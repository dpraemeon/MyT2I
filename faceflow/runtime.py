import contextlib
import copy
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def detect_platform(platform="auto"):
    if platform != "auto":
        return platform
    if "metax" in str(torch.__version__).lower():
        return "metax"
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name().lower()
    return "metax" if "metax" in name else "nvidia"


def prepare_environment(profile):
    # 只补默认环境项，保留启动命令里已经显式设置的值。
    for name, value in profile.get("environment", {}).items():
        os.environ.setdefault(name, str(value))


def setup_distributed(platform="auto", communication="nccl"):
    # torchrun 会提供这些变量；直接启动脚本时按单进程处理。
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    platform = detect_platform(platform)
    if world_size > 1:
        dist.init_process_group("gloo" if communication == "gloo_cpu" else "nccl")
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "platform": platform,
        "communication": communication,
        "collective_device": torch.device("cpu") if communication == "gloo_cpu" else device,
    }


def wrap_distributed(network, env):
    # CPU 通信路径在累积完梯度后手动归约，不交给 DDP 注册同步钩子。
    if env["world_size"] == 1 or env["communication"] == "gloo_cpu":
        return network
    return DistributedDataParallel(
        network,
        device_ids=[env["local_rank"]] if env["device"].type == "cuda" else None,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )


@torch.no_grad()
def synchronize_gradients(model, env):
    if env["world_size"] == 1 or env["communication"] != "gloo_cpu":
        return
    gradients = [
        parameter.grad for parameter in unwrap(model).parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    # 合并成一次 CPU 归约，再按原顺序写回各参数，减少逐参数通信的开销。
    flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
    host = flat.to(device="cpu", non_blocking=False)
    dist.all_reduce(host)
    host.div_(env["world_size"])
    flat.copy_(host, non_blocking=False)
    offset = 0
    for gradient in gradients:
        size = gradient.numel()
        gradient.copy_(flat[offset:offset + size].view_as(gradient))
        offset += size


def autocast_context(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(device, precision):
    # 仅 CUDA 接口下的 FP16 启用梯度缩放，BF16 和 FP32 直接更新。
    return torch.amp.GradScaler(device.type, enabled=precision == "fp16" and device.type == "cuda")


def unwrap(model):
    # DDP 和编译包装可能嵌套，取到原模型后再保存权重或划分参数。
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def create_ema(model):
    # EMA 独立保留 FP32 副本，只用于平滑权重，不参与反向传播。
    return copy.deepcopy(unwrap(model)).float().eval().requires_grad_(False)


@torch.no_grad()
def update_ema(ema, model, decay):
    source = dict(unwrap(model).named_parameters())
    target = dict(ema.named_parameters())
    # 当前阶段冻结的参数无需平滑，缓冲区则直接跟随主模型。
    names = [name for name, parameter in source.items() if parameter.requires_grad]
    torch._foreach_mul_([target[name] for name in names], decay)
    torch._foreach_add_(
        [target[name] for name in names],
        [source[name].detach() for name in names],
        alpha=1.0 - decay,
    )
    for name, value in unwrap(model).named_buffers():
        dict(ema.named_buffers())[name].copy_(value)


def _parameter_group(model, conditioning, learning_rate):
    parameters = [
        (name, parameter) for name, parameter in unwrap(model).named_parameters()
        if unwrap(model).is_conditioning_parameter(name) == conditioning
    ]
    return {
        "params": [parameter for _, parameter in parameters],
        "param_names": [name for name, _ in parameters],
        "name": "conditioning" if conditioning else "main",
        "lr": learning_rate,
        "base_lr": learning_rate,
    }


def build_optimizer(model, training, stage, foreach=False):
    model = unwrap(model)
    model.set_stage(stage)
    learning_rate = training["finetune_lr"] if stage == "text" else training["lr"]
    groups = [_parameter_group(model, False, learning_rate)]
    if stage == "text":
        # 主干使用较小的微调学习率，新接入的文本参数单独设学习率。
        groups.append(_parameter_group(model, True, training["text_lr"]))
    return torch.optim.AdamW(
        groups,
        betas=tuple(training["betas"]),
        weight_decay=training["weight_decay"],
        foreach=foreach,
        fused=False,
    )


def capture_rng_state(device):
    numpy_state = np.random.get_state()
    # NumPy 状态中的数组转为普通列表，便于用 weights_only 模式读取检查点。
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "platform": detect_platform(),
        "torch_version": str(torch.__version__),
    }


def gather_rng_states(device):
    # 各进程的噪声和时间采样不同，断点里需要分别保留它们的随机状态。
    local = capture_rng_state(device)
    if not dist.is_initialized():
        return [local]
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, local)
    return records


def restore_rng_state(record, device, compatible=True):
    random.setstate(record["python"])
    state = record["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(record["torch"])
    if device.type == "cuda":
        # GPU 随机状态只在兼容运行时中恢复，跨平台时按种子重新初始化。
        same_runtime = (
            compatible
            and record["platform"] == detect_platform()
            and record["torch_version"] == str(torch.__version__)
        )
        if same_runtime and record["cuda"] is not None:
            torch.cuda.set_rng_state(record["cuda"], device)
        else:
            torch.cuda.manual_seed(torch.initial_seed())


def save_checkpoint(
    path, model, ema, optimizer, scaler, config, stage, step, stage_step,
    position, runtime, rng_records=None,
):
    model = unwrap(model)
    device = next(model.parameters()).device
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "stage": stage,
        "step": step,
        "stage_step": stage_step,
        "position": position,
        "runtime": runtime,
        "rng_states": rng_records if rng_records is not None else [capture_rng_state(device)],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先完整写入临时文件再替换，读取方不会撞上只保存了一半的检查点。
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def restore_training(
    checkpoint, model, optimizer, scaler, ema, stage, training, foreach=False,
):
    model = unwrap(model)
    saved_stage = checkpoint["stage"]
    if saved_stage == "text" and stage != "text":
        raise ValueError("Resume a text checkpoint with --stage text.")
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    # 先按保存时的阶段重建参数组，确保 Adam 的动量状态能对应到原参数。
    optimizer = build_optimizer(model, training, saved_stage, foreach=foreach)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for group in optimizer.param_groups:
        group["foreach"] = foreach
        group["fused"] = False
    if scaler.is_enabled() and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    stage_step = checkpoint["stage_step"]
    if saved_stage != stage:
        # 转入文本阶段时保留主干的 Adam 状态，只新增条件参数组并重置阶段步数。
        model.set_stage(stage)
        optimizer.param_groups[0]["lr"] = training["finetune_lr"]
        optimizer.param_groups[0]["base_lr"] = training["finetune_lr"]
        optimizer.add_param_group(_parameter_group(model, True, training["text_lr"]))
        with torch.no_grad():
            # 条件分支此前没有训练，EMA 从当前参数起步，主干的平滑历史继续保留。
            current = dict(model.named_parameters())
            for name, parameter in ema.named_parameters():
                if model.is_conditioning_parameter(name):
                    parameter.copy_(current[name])
        stage_step = 0
    model.set_stage(stage)
    ema.set_stage(stage).requires_grad_(False)
    position = dict(checkpoint["position"])
    position.update(step=checkpoint["step"], stage_step=stage_step)
    position.setdefault("epoch", 0)
    position.setdefault("offset", 0)
    return optimizer, position
