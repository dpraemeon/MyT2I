import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .flow import sample
from .model import FaceFlowTiny
from .runtime import autocast_context, detect_platform, load_checkpoint, prepare_environment


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae", help="Override the VAE recorded in the checkpoint.")
    parser.add_argument("--text-encoder", help="Override the checkpoint text encoder.")
    parser.add_argument("--prompt", nargs="+")
    parser.add_argument("--prompts-file")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--method", choices=["euler", "heun"], default="heun")
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--noise-bank", help="Use fixed initial noise from a NumPy array.")
    parser.add_argument("--memory-limit-gib", type=float, help="Limit this process's GPU caching allocator.")
    parser.add_argument("--min-free-gib", type=float, default=0, help="Exit with code 75 if less GPU memory is free.")
    parser.add_argument("--platform", choices=["auto", "nvidia", "metax"], default="auto")
    args = parser.parse_args()
    if args.count < 1 or args.columns < 1 or args.batch_size < 1:
        parser.error("count, columns and batch-size must be positive")
    return args


def save_images(images, output, columns=8):
    output = Path(output)
    assert columns > 0
    columns = min(columns, len(images))
    rows = math.ceil(len(images) / columns)
    width, height = images[0].size
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image, ((index % columns) * width, (index // columns) * height))
    # 图片路径只存网格；目录路径额外保留每张原图。
    if output.suffix.lower() in IMAGE_SUFFIXES:
        output.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output)
    else:
        output.mkdir(parents=True, exist_ok=True)
        grid.save(output / "grid.png")
        for index, image in enumerate(images):
            image.save(output / f"{index:04d}.png")


def initial_noise(count, shape, device, seed, noise_bank=None):
    if noise_bank:
        # 固定噪声库便于跨 checkpoint 对照，哈希用于核对是否为同一份文件。
        path = Path(noise_bank)
        bank = np.load(path, allow_pickle=False)
        assert len(bank) >= count, "Noise bank has fewer entries than --count"
        assert tuple(bank.shape[1:]) == tuple(shape), "Noise bank latent shape differs from the model"
        noise = torch.from_numpy(bank[:count].astype(np.float32, copy=False)).to(device)
        return noise, hashlib.sha256(path.read_bytes()).hexdigest()
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(count, *shape, generator=generator, device=device), None


@torch.inference_mode()
def main():
    args = arguments()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    components = config.get("components", {})
    # 优先沿用训练时记录的组件，命令行可覆盖为迁移后的本地目录。
    vae_id = args.vae or components.get("vae", "stabilityai/sd-vae-ft-mse")
    text_encoder_id = args.text_encoder or components.get("text_encoder", "google/flan-t5-small")
    platform = detect_platform(args.platform)
    profile = config["runtime"].get(platform, {"precision": "fp32", "attention": "math"})
    prepare_environment(profile)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        # 返回 75 表示暂缓采样，记录器会保留任务并稍后重试。
        if free < args.min_free_gib * 2**30:
            print(json.dumps({
                "event": "sampling_deferred", "free_gib": free / 2**30,
                "min_free_gib": args.min_free_gib, "device": str(device),
            }), flush=True)
            raise SystemExit(75)
        if args.memory_limit_gib is not None:
            torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 2**30 / total, device)
    if checkpoint["runtime"].get("platform") == platform:
        profile = checkpoint["runtime"]
    precision = profile["precision"] if device.type == "cuda" else "fp32"
    model = FaceFlowTiny(**config["model"]).to(device)
    model.set_stage(checkpoint["stage"]).set_attention(profile["attention"])
    # 展示和记录统一使用 EMA 权重。
    model.load_state_dict(checkpoint["ema"])
    model.eval().requires_grad_(False)

    prompts = args.prompt or []
    if args.prompts_file:
        prompts = [
            line.strip() for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if prompts and checkpoint["stage"] != "text":
        raise ValueError("Text prompts require a text-stage checkpoint.")
    context, mask = None, None
    if prompts:
        from transformers import AutoTokenizer, T5EncoderModel

        # 提示词不足时按顺序循环，直到覆盖本次要生成的图像数。
        prompts = [prompts[index % len(prompts)] for index in range(args.count)]
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_id)
        encoder = T5EncoderModel.from_pretrained(text_encoder_id).to(device).eval()
        encoded = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=64, return_tensors="pt",
        ).to(device)
        with autocast_context(device, precision):
            context = encoder(**encoded).last_hidden_state
        mask = encoded.attention_mask.bool()
        del encoder

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval().requires_grad_(False)
    noise, noise_sha256 = initial_noise(
        args.count,
        (config["model"]["latent_channels"], config["model"]["image_size"] // 8,
         config["model"]["image_size"] // 8),
        device, args.seed, args.noise_bank,
    )
    images = []
    # 噪声一次生成，再拆成微批；改采样批量不会重新抽取初始噪声。
    for start in range(0, args.count, args.batch_size):
        end = start + args.batch_size
        with autocast_context(device, precision):
            latents = sample(
                model, noise[start:end], steps=args.steps, method=args.method,
                context=context[start:end] if context is not None else None,
                mask=mask[start:end] if mask is not None else None,
                cfg_scale=args.cfg_scale,
            )
        # 还原缓存时的缩放后，再交给冻结的 VAE 解码。
        decoded = vae.decode(latents.float() / 0.18215).sample
        pixels = ((decoded.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        for pixels_one in pixels.permute(0, 2, 3, 1).cpu().numpy():
            images.append(Image.fromarray(np.asarray(pixels_one)))
    save_images(images, args.output, columns=args.columns)
    columns = min(args.columns, len(images))
    # 记录实际采样设置；没有提示词时不启用 CFG。
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output": str(Path(args.output).resolve()), "count": args.count,
        "step": checkpoint["step"], "stage": checkpoint["stage"],
        "seen": checkpoint.get("position", {}).get("seen", 0),
        "seed": args.seed, "method": args.method, "steps": args.steps,
        "columns": columns, "rows": math.ceil(args.count / columns),
        "resolution": config["model"]["image_size"], "batch_size": args.batch_size,
        "noise_bank": str(Path(args.noise_bank).resolve()) if args.noise_bank else None,
        "noise_sha256": noise_sha256,
        "velocity_evaluations": args.steps * (2 if args.method == "heun" else 1),
        "cfg_scale": args.cfg_scale if prompts else 1.0, "platform": platform,
    }
    output = Path(args.output)
    metadata_path = output.with_suffix(".json") if output.suffix.lower() in IMAGE_SUFFIXES else output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError:
        # 和空闲显存不足使用同一返回码，供记录器延期处理。
        print(json.dumps({"event": "sampling_deferred", "reason": "allocator_oom"}), flush=True)
        raise SystemExit(75)
