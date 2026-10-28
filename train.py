import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from faceflow.data import load_cache, VAE_ID, SCALING_FACTOR
from faceflow.flow import rf_loss, sample
from faceflow.model import FaceFlowTiny
from faceflow.runtime import (
    setup_distributed, autocast_context, make_scaler, create_ema, update_ema,
    build_optimizer, restore_training, save_checkpoint, load_checkpoint,
    gather_rng_states, restore_rng_state,
    detect_platform, prepare_environment, wrap_distributed,
    synchronize_gradients,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--stage", choices=["unconditional", "text"], default="unconditional")
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--platform", choices=["auto", "nvidia", "metax"], default="auto")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--steps", type=int, default=0, help="Additional successful updates; 0 uses the time budget.")
    parser.add_argument("--include-prepare-time", action="store_true")
    parser.add_argument("--save-every", type=float, default=3600)
    parser.add_argument("--no-preview", action="store_true", help="Save checkpoints without loading the VAE.")
    parser.add_argument("--vae", help="Override the VAE recorded by the latent cache.")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--synthetic-count", type=int, default=162770)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--attention", choices=["math", "sdpa"])
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--global-batch", type=int, help="Override the platform or shared global batch.")
    parser.add_argument("--cache-device", choices=["cpu", "cuda"])
    parser.add_argument("--foreach", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if not args.benchmark and not args.cache:
        parser.error("--cache is required outside benchmark mode")
    return args


def fetch_batch(cache, indices, device):
    # 潜变量可能常驻 GPU；文本表和描述索引仍在 CPU 上取数。
    latent_indices = indices.to(cache["latents"].device)
    z = cache["latents"].index_select(0, latent_indices).to(device, non_blocking=True)
    context, mask = None, None
    if cache["context"] is not None:
        caption_ids = cache["caption_ids"].index_select(0, indices)
        context = cache["context"].index_select(0, caption_ids).to(device, non_blocking=True)
        mask = cache["mask"].index_select(0, caption_ids).to(device, non_blocking=True)
    return z, context, mask


@torch.no_grad()
def preview(model, output, device, precision, vae_id, count=16):
    # 出图后恢复随机数状态，避免预览改变后续训练的随机序列。
    with torch.random.fork_rng(devices=[device.index]):
        _render_preview(model, output, device, precision, vae_id, count)


@torch.no_grad()
def _render_preview(model, output, device, precision, vae_id, count):
    from diffusers import AutoencoderKL
    from PIL import Image

    # 每次从同一批噪声出发，便于比较训练前后的变化。
    generator = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(count, 4, 16, 16, device=device, generator=generator)
    with autocast_context(device, precision):
        latents = sample(model, noise, steps=32)
    vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval()
    images = []
    # VAE 按小批量解码，控制保存预览时的额外显存占用。
    for latent in latents.split(4):
        decoded = vae.decode(latent.float() / SCALING_FACTOR).sample
        pixels = ((decoded.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
        images.extend(Image.fromarray(image) for image in pixels)
    columns = 4
    grid = Image.new("RGB", (columns * 128, ((count + columns - 1) // columns) * 128))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * 128, (index // columns) * 128))
    grid.save(output)
    del vae
    torch.cuda.empty_cache()


def main():
    args = arguments()
    run_started = time.monotonic()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    platform = detect_platform(args.platform)
    assert platform in config["runtime"], "Install a CUDA or MACA PyTorch GPU environment before training"
    # 平台配置先应用阶段覆盖，后面的命令行参数拥有最高优先级。
    runtime = dict(config["runtime"][platform])
    runtime.update(runtime.get("stages", {}).get(args.stage, {}))
    prepare_environment(runtime)
    env = setup_distributed(platform, communication=runtime.get("communication", "nccl"))
    rank, world, device = env["rank"], env["world_size"], env["device"]
    collective_device = env["collective_device"]
    assert device.type == "cuda", "Training requires a CUDA or MACA PyTorch GPU environment"
    for key in ("precision", "attention", "micro_batch", "global_batch", "cache_device", "foreach", "compile"):
        value = getattr(args, key)
        if value is not None:
            runtime[key] = value
    torch.set_num_threads(runtime.get("cpu_threads", 4))
    micro = runtime["micro_batch"]
    global_batch = runtime.get("global_batch", training["global_batch"])
    training["global_batch"] = global_batch
    runtime["global_batch"] = global_batch
    # 一次更新覆盖所有卡的微批，剩余部分通过梯度累积补齐。
    assert micro > 0 and global_batch > 0 and global_batch % (world * micro) == 0, \
        "global_batch must be a positive multiple of world_size * micro_batch"
    accumulation = global_batch // (world * micro)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    # 各卡先用相同种子初始化模型，再分开训练中的随机序列。
    torch.manual_seed(seed)
    model = FaceFlowTiny(**config["model"]).to(device)
    model.set_stage(args.stage)
    model.set_attention(runtime["attention"])
    ema = create_ema(model)
    optimizer = build_optimizer(model, training, args.stage, foreach=runtime["foreach"])
    scaler = make_scaler(device, runtime["precision"])
    position = {"epoch": 0, "offset": 0, "seen": 0}
    step, stage_step = 0, 0
    torch.manual_seed(seed + rank)
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        assert checkpoint["config"]["model"] == config["model"], "Model config differs from checkpoint"
        optimizer, saved = restore_training(
            checkpoint, model, optimizer, scaler, ema, args.stage, training,
            foreach=runtime["foreach"],
        )
        position = {key: saved.get(key, 0) for key in ("epoch", "offset", "seen")}
        step = checkpoint["step"]
        stage_step = saved["stage_step"]
        saved_runtime = checkpoint["runtime"]
        # 运行环境兼容时恢复各卡随机状态；换平台或卡数则重建设备随机流。
        compatible = (
            saved_runtime.get("platform") == env["platform"]
            and saved_runtime.get("torch_version") == str(torch.__version__)
            and len(checkpoint["rng_states"]) == world
        )
        torch.manual_seed(seed + rank + step)
        if checkpoint["rng_states"]:
            restore_rng_state(
                checkpoint["rng_states"][rank % len(checkpoint["rng_states"])],
                device, compatible=compatible,
            )
        if not compatible:
            torch.cuda.manual_seed(seed + rank + step)
    ema.set_stage(args.stage)
    ema.requires_grad_(False)
    ema.set_attention(runtime["attention"])
    cache = load_cache(
        args.cache, device=device if runtime["cache_device"] == "cuda" else "cpu",
        synthetic_count=args.synthetic_count if args.benchmark else 0, seed=seed,
    )
    components = config.setdefault("components", {})
    if not args.benchmark:
        # 把缓存实际使用的组件写进 checkpoint，采样时据此加载。
        components.update(
            vae=cache["metadata"]["vae_id"],
            text_encoder=cache["metadata"].get("text_encoder", "google/flan-t5-small"),
            scaling_factor=cache["metadata"]["scaling_factor"],
        )
    vae_id = args.vae or components.get("vae", VAE_ID)
    if args.stage == "text" and args.benchmark:
        # 测速使用合成文本，不需要下载编码器或准备真实属性。
        generator = torch.Generator().manual_seed(seed + 7)
        cache["context"] = torch.randn(64, 64, 512, generator=generator, dtype=torch.float16)
        cache["mask"] = torch.arange(64)[None, :].expand(64, -1) < 48
        cache["caption_ids"] = torch.arange(len(cache["latents"])) % 64
    assert args.stage != "text" or cache["context"] is not None, "Run prepare.py --text before text training"
    count = len(cache["latents"])
    assert count >= global_batch
    runtime.update(platform=env["platform"], torch_version=str(torch.__version__))
    network = torch.compile(model) if runtime["compile"] else model
    network = wrap_distributed(network, env)
    network.train()
    target_updates = (args.steps or 50) + args.warmup if args.benchmark else args.steps
    completed = 0
    start = time.monotonic()
    deadline = run_started + args.hours * 3600
    if args.include_prepare_time and not args.benchmark:
        deadline = time.monotonic() + args.hours * 3600 - max(
            0, time.time() - cache["metadata"]["prepare_started_at"],
        )
    # 按时间训练时预留半小时保存和出图；指定步数仍受总时限约束。
    train_deadline = deadline if args.benchmark or args.steps else deadline - 1800
    last_save = start
    last_preview_step = -1
    timed_start = None
    timed_updates = 0
    measured_loss = 0.0
    epoch = position["epoch"]
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
    torch.cuda.reset_peak_memory_stats(device)
    if world > 1:
        dist.barrier()
    if args.benchmark and args.warmup == 0:
        timed_start = time.monotonic()
    if rank == 0:
        print(json.dumps({
            "event": "start", "platform": env["platform"], "stage": args.stage,
            "parameters": sum(p.numel() for p in model.parameters()), "runtime": runtime,
            "accumulation": accumulation, "world_size": world,
        }), flush=True)

    while completed < target_updates or target_updates == 0:
        # 尾部不足一个全局 batch 就进入下一轮，各卡取同一排列中的不同片段。
        if position["offset"] + global_batch > count:
            epoch += 1
            position.update(epoch=epoch, offset=0)
            permutation = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch))
        local_count = global_batch // world
        left = position["offset"] + rank * local_count
        indices = permutation[left:left + local_count]
        position["offset"] += global_batch
        position["seen"] = position.get("seen", 0) + global_batch
        optimizer.zero_grad(set_to_none=True)
        # 切换到文本阶段后，学习率预热按该阶段的更新数重新计算。
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * min(1.0, (stage_step + 1) / training["warmup_steps"])
        update_loss = torch.zeros((), device=device)
        for micro_index in range(accumulation):
            selected = indices[micro_index * micro:(micro_index + 1) * micro]
            z, context, mask = fetch_batch(cache, selected, device)
            drop_mask = None
            if args.stage == "text":
                drop_mask = torch.rand(len(selected), device=device) < training["condition_dropout"]
            # DDP 只在最后一个微批归约；CPU/Gloo 路径在累积完成后统一同步。
            sync = (
                network.no_sync()
                if world > 1 and env["communication"] == "nccl" and micro_index < accumulation - 1
                else nullcontext()
            )
            with sync:
                with autocast_context(device, runtime["precision"]):
                    loss = rf_loss(network, z, context, mask, drop_mask) / accumulation
                scaler.scale(loss).backward()
            update_loss += loss.detach()
        synchronize_gradients(model, env)
        # 梯度先解除缩放再裁剪，阈值才不受动态精度缩放影响。
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # FP16 溢出会跳过优化器更新，此时不推进成功步数和 EMA。
        success = scaler.get_scale() >= old_scale
        if not scaler.is_enabled():
            assert torch.isfinite(grad_norm), "Non-finite gradient"
        if success:
            update_ema(ema, model, training["ema_decay"])
            step += 1
            stage_step += 1
            completed += 1
            if args.benchmark and completed > args.warmup:
                timed_updates += 1
                measured_loss += update_loss.item()
        if args.benchmark and timed_start is None and completed >= args.warmup:
            # 预热和编译开销单独记录，不计入稳定阶段的吞吐。
            torch.cuda.synchronize(device)
            if world > 1:
                dist.barrier()
            timed_start = time.monotonic()
        if completed % 10 == 0 and success:
            if rank == 0:
                print(json.dumps({
                    "event": "update", "step": step, "loss": update_loss.item(),
                    "elapsed_s": round(time.monotonic() - start, 2),
                }), flush=True)
        if args.benchmark:
            stop = torch.tensor(int(time.monotonic() >= deadline), device=collective_device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
            continue
        control = torch.tensor([
            int(time.monotonic() - last_save >= args.save_every),
            int(time.monotonic() >= train_deadline),
        ], device=collective_device)
        if world > 1:
            # 由主卡统一决定保存和停止，避免各卡进入不同的同步分支。
            dist.broadcast(control, src=0)
        should_save, should_stop = control.tolist()
        if should_save or should_stop:
            # 随机状态需要各卡一起收集，文件和预览只由主卡生成。
            rng_records = gather_rng_states(device)
            if rank == 0:
                save_checkpoint(
                    output / "latest.pt", model, ema, optimizer, scaler, config,
                    args.stage, step, stage_step, position, runtime, rng_records,
                )
                if not args.no_preview:
                    preview(ema, output / f"preview-{step:07d}.png", device, runtime["precision"], vae_id)
                    last_preview_step = step
            if world > 1:
                dist.barrier()
            last_save = time.monotonic()
        if should_stop:
            break

    torch.cuda.synchronize(device)
    if args.benchmark:
        elapsed = time.monotonic() - (timed_start or start)
        statistics = torch.tensor([
            elapsed, torch.cuda.max_memory_allocated(device) / 2**30,
            torch.cuda.max_memory_reserved(device) / 2**30,
        ], device=collective_device, dtype=torch.float64)
        if world > 1:
            # 多卡耗时和两类显存峰值分别取最大值。
            dist.all_reduce(statistics, op=dist.ReduceOp.MAX)
        if rank == 0:
            result = {
                "runtime": runtime, "world_size": world, "global_batch": global_batch,
                "measured_updates": timed_updates, "elapsed_s": statistics[0].item(),
                "samples_per_second": timed_updates * global_batch / statistics[0].item(),
                "peak_allocated_gib": statistics[1].item(),
                "peak_reserved_gib": statistics[2].item(),
                "loss": measured_loss / max(1, timed_updates),
                "startup_and_warmup_s": (timed_start or start) - start,
                "device": torch.cuda.get_device_name(device),
            }
            (output / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"event": "benchmark", **result}), flush=True)
    else:
        # 正常退出前再保存一次，覆盖最后一段尚未定时保存的更新。
        rng_records = gather_rng_states(device)
        if rank == 0:
            save_checkpoint(
                output / "latest.pt", model, ema, optimizer, scaler, config,
                args.stage, step, stage_step, position, runtime, rng_records,
            )
            if not args.no_preview and last_preview_step != step:
                preview(ema, output / f"preview-{step:07d}.png", device, runtime["precision"], vae_id)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
