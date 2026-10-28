import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch


DEFAULTS = dict(every_hours=5, count=32, columns=8, seed=42, steps=40,
                method="heun", batch_size=2, device=0, memory_limit_gib=3,
                min_free_gib=4, poll_seconds=30)
SERIES_KEYS = ("count", "columns", "seed", "steps", "method", "batch_size")


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def write_json(path, value):
    # 先写临时文件再替换，避免中断后留下半份索引。
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def lock_recorder(directory):
    # 同一个记录目录只允许一个进程维护，避免重复归档和覆盖索引。
    handle = (directory / ".recorder.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def process_alive(pid):
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_uint)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint))
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_uint()
        kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        return code.value == 259
    if Path("/proc").is_dir():
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, PermissionError):
            state = None
        # 僵尸进程仍有 PID，但已经不能继续训练。
        if state == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_updates(log):
    updates = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        # 启动日志可能夹着警告或编译信息，只提取训练更新事件。
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "update":
            updates.append(record)
    return updates


def save_history(directory, updates):
    temporary = directory / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "elapsed_s", "loss"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in updates)
    temporary.replace(directory / "history.csv")


def prepare_series(directory, config, settings):
    size = config["model"]["image_size"] // 8
    shape = (settings["count"], config["model"]["latent_channels"], size, size)
    series = {key: settings[key] for key in SERIES_KEYS}
    series["noise_shape"] = list(shape)
    bank = directory / "noise.npy"
    # 一组里程碑共用同一批噪声和采样设置，续开记录器时也要核对。
    if bank.exists():
        assert np.load(bank, mmap_mode="r").shape == shape, "Noise bank shape differs"
    else:
        noise = np.random.default_rng(settings["seed"]).standard_normal(shape).astype(np.float32)
        np.save(bank, noise)
    series["noise_sha256"] = hashlib.sha256(bank.read_bytes()).hexdigest()
    path = directory / "settings.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == series, \
            "Comparison settings changed; use a new records directory for a new series."
    else:
        write_json(path, series)
    return bank


def copy_snapshot(source, destination):
    if os.name != "nt":
        shutil.copyfile(source, destination)
        return
    # Windows 句柄允许共享删除，读取期间训练器仍能原子替换 latest.pt。
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.c_void_p)
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(source), 0x80000000, 7, None, 3, 128, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    with os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY), "rb") as src, \
            destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)


def capture_checkpoint(run, directory, index, updates, kind, target=None):
    # 先复制完整快照，后续读取和出图不再依赖持续变化的 latest.pt。
    temporary = directory / ".pending.pt"
    copy_snapshot(run / "latest.pt", temporary)
    checkpoint = torch.load(temporary, map_location="cpu", weights_only=True, mmap=True)
    step = int(checkpoint["step"])
    stage = checkpoint["stage"]
    seen = int(checkpoint["position"].get("seen", 0))
    del checkpoint
    gc.collect()
    # 取该权重之前最近一条更新日志，只把它当作训练时长的保守下界。
    preceding = [row for row in updates if row["step"] <= step]
    elapsed = preceding[-1]["elapsed_s"] if preceding else 0.0
    if target is not None and elapsed < target * 3600:
        temporary.unlink()
        return None
    same = next((entry for entry in index["entries"]
                 if entry["step"] == step and entry["kind"] == kind
                 and entry["target_hours"] == target), None)
    if same is not None:
        temporary.unlink()
        return same
    name = f"{kind}-h{elapsed / 3600:06.3f}-step{step:07d}"
    snapshot = directory / f"{name}.pt"
    temporary.replace(snapshot)
    entry = dict(kind=kind, target_hours=target, elapsed_s=elapsed, step=step,
                 seen=seen, stage=stage, checkpoint=snapshot.name,
                 grid=f"{name}.png", status="pending",
                 captured_at=datetime.now(timezone.utc).isoformat(),
                 elapsed_source="preceding logged update; conservative lower bound")
    if kind == "final":
        # final 表示训练进程已结束，不把最后保存的权重标成精确预算时刻。
        entry["reason"] = "training_process_ended"
    index["entries"].append(entry)
    write_json(directory / "index.json", index)
    emit("checkpoint_archived", **entry)
    return entry


def render_pending(directory, index, settings, bank):
    expected_noise = json.loads((directory / "settings.json").read_text(encoding="utf-8"))["noise_sha256"]
    for entry in index["entries"]:
        if entry["status"] == "complete":
            continue
        command = [sys.executable, "-m", "faceflow.sample",
                   "--checkpoint", str(directory / entry["checkpoint"]),
                   "--output", str(directory / entry["grid"]),
                   "--noise-bank", str(bank), "--platform", "auto"]
        for flag in ("count", "columns", "seed", "steps", "method", "batch_size",
                     "memory_limit_gib", "min_free_gib"):
            command.extend(["--" + flag.replace("_", "-"), str(settings[flag])])
        # 采样独立启动，显存限制和目标设备不会改动正在运行的训练器。
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(settings["device"]))
        result = subprocess.run(command, env=environment)
        if result.returncode == 75:
            # 快照已归档，显存不足时保留 pending 状态，下一轮再尝试出图。
            emit("render_deferred", step=entry["step"], grid=entry["grid"])
            return False
        result.check_returncode()
        metadata_path = (directory / entry["grid"]).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["step"] == entry["step"], "Rendered checkpoint differs from archive"
        assert metadata["noise_sha256"] == expected_noise, "Comparison noise bank changed"
        entry.update(status="complete", metadata=metadata_path.name,
                     noise_sha256=metadata["noise_sha256"])
        write_json(directory / "index.json", index)
        make_comparison(directory, index)
        emit("grid_saved", step=entry["step"], grid=entry["grid"])
    return True


def make_comparison(directory, index):
    # 标题使用日志记录的时长和 step，而不是目标里程碑小时数。
    entries = sorted((entry for entry in index["entries"] if entry["status"] == "complete"),
                     key=lambda entry: (entry["elapsed_s"], entry["step"]))
    if not entries:
        return
    with Image.open(directory / entries[0]["grid"]) as first:
        width, height = first.size
    comparison = Image.new("RGB", (len(entries) * width, height + 40), "white")
    draw = ImageDraw.Draw(comparison)
    for column, entry in enumerate(entries):
        with Image.open(directory / entry["grid"]) as grid:
            assert grid.size == (width, height), "Comparison grid dimensions changed"
            comparison.paste(grid.convert("RGB"), (column * width, 40))
        draw.text((column * width + 8, 6),
                  f"{entry['elapsed_s'] / 3600:.2f} h | step {entry['step']} | {entry['kind']}",
                  fill="black")
    temporary = directory / "comparison.tmp.png"
    comparison.save(temporary)
    temporary.replace(directory / "comparison.png")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.json"))
    parser.add_argument("--start-hour", type=float, default=10)
    parser.add_argument("--every-hours", type=float)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--capture-now", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    run, log = Path(args.run).resolve(), Path(args.log).resolve()
    directory = run / "records"
    directory.mkdir(exist_ok=True)
    lock = lock_recorder(directory)
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        settings = {**DEFAULTS, **config.get("recording", {})}
        if args.every_hours is not None:
            settings["every_hours"] = args.every_hours
        assert args.start_hour >= 0 and settings["every_hours"] > 0
        assert settings["count"] > 0 and settings["columns"] > 0
        assert settings["poll_seconds"] > 0
        bank = prepare_series(directory, config, settings)
        path = directory / "index.json"
        index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
        pid = args.pid
        if not args.once:
            if pid is None:
                pid_file = run / "launcher.pid"
                assert pid_file.exists(), "Watching requires --pid or run/launcher.pid"
                pid = int(pid_file.read_text().strip())
            assert pid > 0
        checked = None
        emit("recorder_start", run=str(run), start_hour=args.start_hour,
             every_hours=settings["every_hours"], count=settings["count"],
             columns=settings["columns"], pid=pid)
        while True:
            updates = read_updates(log)
            save_history(directory, updates)
            # 用进程存活情况识别结束，不能只看日志是否暂时停止增长。
            ended = not args.once and not process_alive(pid)
            source = run / "latest.pt"
            if source.exists():
                signature = (source.stat().st_ino, source.stat().st_size, source.stat().st_mtime_ns)
                baseline = args.once or (args.capture_now and not any(
                    entry["kind"] == "baseline" for entry in index["entries"]))
                targets = {entry["target_hours"] for entry in index["entries"]
                           if entry["kind"] == "milestone"}
                target = args.start_hour
                while target in targets:
                    target += settings["every_hours"]
                elapsed = updates[-1]["elapsed_s"] if updates else 0.0
                due = elapsed >= target * 3600
                if baseline:
                    capture_checkpoint(run, directory, index, updates, "baseline")
                    checked = signature
                if signature != checked and due:
                    # 到点后等待符合时长的已保存权重，不读取训练器内存中的模型。
                    capture_checkpoint(run, directory, index, updates, "milestone", target)
                    checked = signature
                if ended and not any(entry["kind"] == "final" for entry in index["entries"]):
                    capture_checkpoint(run, directory, index, updates, "final")
            completed = render_pending(directory, index, settings, bank)
            if args.once:
                return 0 if completed else 75
            if ended and completed:
                emit("recorder_finished", reason="training_process_ended")
                return 0
            time.sleep(settings["poll_seconds"])
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
