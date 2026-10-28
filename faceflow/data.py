import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


VAE_ID = "stabilityai/sd-vae-ft-mse"
TEXT_ENCODER_ID = "google/flan-t5-small"
SCALING_FACTOR = 0.18215


def celeba_filenames(partition, split="train"):
    # 沿用官方划分和文件顺序，后续图像潜变量与文本索引都按这个顺序对齐。
    split_id = {"train": "0", "val": "1", "test": "2"}[split]
    with open(partition, encoding="utf-8") as stream:
        return [name for name, part in map(str.split, stream) if part == split_id]


def read_attributes(path):
    with open(path, encoding="utf-8") as stream:
        # 属性文件前两行分别是样本数和列名，真正的数据从第三行开始。
        next(stream)
        names = next(stream).split()
        return {
            parts[0]: dict(zip(names, map(int, parts[1:])))
            for parts in map(str.split, stream)
        }


def attribute_caption(attributes):
    """只描述正标签；发色冲突时不写发色。"""
    bald = attributes["Bald"] > 0
    caption = "A portrait of a bald person" if bald else "A portrait of a person"
    hair = [
        color
        for label, color in (
            ("Black_Hair", "black"),
            ("Blond_Hair", "blond"),
            ("Brown_Hair", "brown"),
        )
        if attributes[label] > 0
    ]
    traits = []
    # 秃头样本不再补发色和刘海，避免一句描述里出现互相矛盾的特征。
    if not bald and len(hair) == 1:
        traits.append(f"{hair[0]} hair")
    if not bald and attributes["Bangs"] > 0:
        traits.append("bangs")
    if traits:
        caption += " with " + " and ".join(traits)
    if attributes["Eyeglasses"] > 0:
        caption += ", wearing eyeglasses"
    if attributes["Smiling"] > 0:
        caption += ", smiling"
    return caption + "."


class CelebAImages(Dataset):
    def __init__(self, images, filenames, resolution=128):
        self.images = Path(images)
        self.filenames = filenames
        self.resolution = resolution

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with Image.open(self.images / self.filenames[index]) as image:
            image = image.convert("RGB")
            width, height = image.size
            # 先居中裁成正方形再缩放，避免直接拉伸改变脸部比例。
            side = min(width, height)
            left, top = (width - side) // 2, (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
            # VAE 接收 [-1, 1] 的像素，返回时把通道维移到最前面。
            pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(pixels).permute(2, 0, 1)


def load_cache(path, device="cpu", synthetic_count=0, latent_shape=(4, 16, 16), seed=42):
    """仅将图像潜变量放到指定设备，去重后的文本特征留在 CPU。"""
    result = {"context": None, "mask": None, "caption_ids": None}
    if synthetic_count:
        # 合成数据只用于测吞吐，不经过图像编码，也不代表真实训练数据。
        generator = torch.Generator().manual_seed(seed)
        result["latents"] = torch.randn(
            synthetic_count, *latent_shape, generator=generator, dtype=torch.float16
        ).to(device)
        result["metadata"] = {
            "synthetic": True,
            "count": synthetic_count,
            "resolution": latent_shape[-1] * 8,
            "latent_shape": list(latent_shape),
            "scaling_factor": SCALING_FACTOR,
            "posterior": "mean",
        }
        return result

    path = Path(path)
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    # 缓存必须与训练时的潜空间约定一致，尤其不能混用采样值和后验均值。
    assert metadata["resolution"] == 128
    assert abs(metadata["scaling_factor"] - SCALING_FACTOR) < 1e-7
    assert metadata["posterior"] == "mean"
    latents = torch.from_numpy(np.load(path / "latents.npy", mmap_mode="c"))
    assert tuple(latents.shape[1:]) == tuple(latent_shape)
    result.update(latents=latents.to(device), metadata=metadata)
    if "text_encoder" in metadata:
        # 多张图可以共享同一条描述；取批次时再通过 caption_ids 找到文本特征。
        result["context"] = torch.from_numpy(np.load(path / "text.npy", mmap_mode="c"))
        result["mask"] = torch.from_numpy(np.load(path / "text_mask.npy", mmap_mode="c"))
        result["caption_ids"] = torch.from_numpy(
            np.load(path / "caption_ids.npy", mmap_mode="c")
        )
    return result
