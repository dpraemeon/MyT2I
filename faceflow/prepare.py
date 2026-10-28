import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import (
    SCALING_FACTOR,
    TEXT_ENCODER_ID,
    VAE_ID,
    CelebAImages,
    attribute_caption,
    celeba_filenames,
    read_attributes,
)
from .runtime import detect_platform, prepare_environment


@torch.inference_mode()
def prepare_latents(args):
    from diffusers import AutoencoderKL
    started = time.time()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    # 重建缓存前移除旧元数据，避免把尚未写完的潜变量当作完整缓存。
    (cache / "meta.json").unlink(missing_ok=True)
    filenames = celeba_filenames(args.partition, args.split)
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device).eval().requires_grad_(False)
    assert vae.config.latent_channels == 4
    assert abs(vae.config.scaling_factor - SCALING_FACTOR) < 1e-7
    dataset = CelebAImages(args.images, filenames)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    latents = np.lib.format.open_memmap(
        cache / "latents.npy", mode="w+", dtype=np.float16, shape=(len(dataset), 4, 16, 16)
    )
    offset = 0
    for pixels in loader:
        # 固定取后验均值；缩放约定要和训练、VAE 解码时保持一致。
        encoded = vae.encode(pixels.to(device, non_blocking=True)).latent_dist.mean
        batch = (encoded * SCALING_FACTOR).cpu().numpy().astype(np.float16)
        latents[offset : offset + len(batch)] = batch
        offset += len(batch)
        if offset % (args.batch_size * 100) == 0:
            print(f"Encoded {offset}/{len(dataset)} images", flush=True)
    # 先写完数组，再保存描述图像顺序和编码约定的元数据。
    latents.flush()
    metadata = {
        "vae_id": args.vae,
        "scaling_factor": SCALING_FACTOR,
        "resolution": 128,
        "posterior": "mean",
        "latent_shape": [4, 16, 16],
        "split": args.split,
        "count": len(filenames),
        "filenames": filenames,
        "prepare_started_at": started,
        "created_at": time.time(),
    }
    (cache / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(filenames)} latents to {cache}", flush=True)


@torch.inference_mode()
def prepare_text(args):
    from transformers import AutoTokenizer, T5EncoderModel

    cache = Path(args.cache)
    metadata = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    attributes = read_attributes(args.attributes)
    # 沿用潜变量的图像顺序，同一句属性描述只需编码一次。
    captions = [attribute_caption(attributes[name]) for name in metadata["filenames"]]
    unique = sorted(set(captions))
    caption_index = {caption: index for index, caption in enumerate(unique)}
    caption_ids = np.array([caption_index[caption] for caption in captions], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    encoder = T5EncoderModel.from_pretrained(args.text_encoder)
    assert encoder.config.d_model == 512
    encoder = encoder.to(args.device).eval().requires_grad_(False)
    # 特征表按去重后的描述存储，caption_ids 负责映射回每张图。
    features = np.lib.format.open_memmap(
        cache / "text.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(unique), args.max_length, encoder.config.d_model),
    )
    masks = np.lib.format.open_memmap(
        cache / "text_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(unique), args.max_length),
    )
    for start in range(0, len(unique), args.batch_size):
        end = min(start + args.batch_size, len(unique))
        tokens = tokenizer(
            unique[start:end],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(args.device)
        hidden = encoder(**tokens).last_hidden_state
        features[start:end] = hidden.cpu().numpy().astype(np.float16)
        # 保留填充位置的掩码，交叉注意力只读取有效文本。
        masks[start:end] = tokens.attention_mask.cpu().numpy().astype(np.bool_)
    features.flush()
    masks.flush()
    np.save(cache / "caption_ids.npy", caption_ids)
    # 文本数组全部落盘后，才在元数据中标记文本缓存可用。
    metadata.update(
        text_encoder=args.text_encoder,
        text_dim=encoder.config.d_model,
        max_length=args.max_length,
        captions=unique,
        text_created_at=time.time(),
    )
    (cache / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(unique)} unique captions for {len(captions)} images", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.json"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--images")
    parser.add_argument("--partition")
    parser.add_argument("--attributes")
    parser.add_argument("--vae", default=VAE_ID)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--text-encoder", default=TEXT_ENCODER_ID)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    profile = json.loads(Path(args.config).read_text(encoding="utf-8"))["runtime"]
    prepare_environment(profile.get(detect_platform(), {}))
    # 图像和文本分开准备，文本阶段直接补充已有的图像缓存。
    if args.text:
        if not args.attributes:
            parser.error("--text requires --attributes")
        prepare_text(args)
    else:
        if not args.images or not args.partition:
            parser.error("latent preparation requires --images and --partition")
        prepare_latents(args)


if __name__ == "__main__":
    main()
