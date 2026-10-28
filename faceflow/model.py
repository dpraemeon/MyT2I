"""在潜空间中生成图像，通过时间门控引入局部混合的轻量 DiT。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        # 归一化单独用 FP32，算完再转回输入精度，减少混合精度下的数值波动。
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight = self.weight.float() if self.weight is not None else None
            bias = self.bias.float() if self.bias is not None else None
            result = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
        return result.to(x.dtype)


def sinusoidal_embedding(values, width):
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(width // 2, device=values.device).float()
        / (width // 2)
    )
    angles = values.float()[..., None] * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def position_embedding(size, width):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    # 横纵坐标各占一半通道，展平顺序与后面的潜变量序列保持一致。
    return torch.cat(
        (sinusoidal_embedding(x.flatten(), width // 2),
         sinusoidal_embedding(y.flatten(), width // 2)), dim=-1
    )[None]


def attend(query, key, value, mask, implementation):
    # 掩码只限制文本键的位置，并广播到所有注意力头和图像查询位置。
    allowed = mask[:, None, None, :] if mask is not None else None
    if implementation == "sdpa":
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False
        )
    # 数学实现显式用 FP32 计算注意力；SDPA 路径交给 PyTorch 选择实现。
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = query.float() @ key.float().transpose(-2, -1)
        logits = logits * (query.shape[-1] ** -0.5)
        if allowed is not None:
            logits = logits.masked_fill(~allowed, float("-inf"))
        result = logits.softmax(dim=-1) @ value.float()
    return result.to(query.dtype)


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.implementation = "math"

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, width // self.heads)
        # 拆成 [批量, 头数, 序列长度, 每头维度]，在各头内独立计算注意力。
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        result = attend(query, key, value, None, self.implementation)
        return self.proj(result.transpose(1, 2).reshape(batch, tokens, width))


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm = FP32LayerNorm(width, eps=1e-6)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.implementation = "math"

    def forward(self, x, context, mask):
        batch, tokens, width = x.shape
        # 图像特征提供查询，文本特征提供键和值，两边序列长度可以不同。
        query = self.query(self.norm(x)).reshape(
            batch, tokens, self.heads, width // self.heads
        ).transpose(1, 2)
        key_value = self.key_value(context).reshape(
            batch, context.shape[1], 2, self.heads, width // self.heads
        )
        key, value = key_value.permute(2, 0, 3, 1, 4).unbind(0)
        result = attend(query, key, value, mask, self.implementation)
        return self.proj(result.transpose(1, 2).reshape(batch, tokens, width))


class LocalFFN(nn.Module):
    def __init__(self, width, hidden, gate_hidden):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.gate = nn.Sequential(
            nn.Linear(width, gate_hidden), nn.SiLU(), nn.Linear(gate_hidden, 1)
        )
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x, time_embedding, height, width):
        expanded = self.fc1(x)
        # 临时还原二维网格，用逐通道卷积补上相邻潜变量之间的局部信息。
        grid = expanded.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        local = self.local(grid).flatten(2).transpose(1, 2)
        # 每个样本得到一个时间门值，在当前层的所有位置和通道上共享。
        gate = self.gate(time_embedding).sigmoid()[:, None]
        return self.fc2(F.silu(expanded + gate * local))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, width, heads, mlp_ratio, gate_hidden, cross_attention):
        super().__init__()
        self.norm1 = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.norm2 = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.attention = SelfAttention(width, heads)
        self.ffn = LocalFFN(width, width * mlp_ratio, gate_hidden)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.cross_attention = CrossAttention(width, heads) if cross_attention else None

    def forward(self, x, time_embedding, context, mask, height, width):
        # 时间条件分别控制注意力和前馈分支的平移、缩放与残差强度。
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(
            time_embedding
        ).chunk(6, dim=-1)
        x = x + gate1[:, None] * self.attention(modulate(self.norm1(x), shift1, scale1))
        if self.cross_attention is not None and context is not None:
            x = x + self.cross_attention(x, context, mask)
        update = self.ffn(
            modulate(self.norm2(x), shift2, scale2), time_embedding, height, width
        )
        return x + gate2[:, None] * update


class FaceFlowTiny(nn.Module):
    def __init__(
        self, image_size=128, latent_channels=4, width=384, depth=12,
        heads=6, mlp_ratio=4, cross_attention_layers=(4, 8, 12),
        text_dim=512, time_dim=256, gate_hidden=32,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.latent_channels = latent_channels
        self.time_dim = time_dim
        self.input_projection = nn.Linear(latent_channels, width)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        # 位置编码可由尺寸重新生成，跟随设备迁移但不写入检查点。
        self.register_buffer(
            "position_embedding", position_embedding(self.latent_size, width),
            persistent=False,
        )
        # 层号从 1 开始，默认只在第 4、8、12 层接入文本交叉注意力。
        self.blocks = nn.ModuleList(
            Block(width, heads, mlp_ratio, gate_hidden, index in cross_attention_layers)
            for index in range(1, depth + 1)
        )
        self.text_norm = FP32LayerNorm(text_dim, eps=1e-6)
        self.text_projection = nn.Linear(text_dim, width)
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        self.final_norm = FP32LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_projection = nn.Linear(width, latent_channels)
        self.initialize_weights()
        self.set_stage("unconditional")

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            # 残差门从零开始，让主干块在初始化时先接近恒等映射。
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
            # 局部分支的门最后过 sigmoid，零初始化对应的门值是 0.5。
            nn.init.zeros_(block.ffn.gate[-1].weight)
            nn.init.zeros_(block.ffn.gate[-1].bias)
            if block.cross_attention is not None:
                # 文本分支先不改变主干输出，之后由训练逐步学会注入条件。
                nn.init.zeros_(block.cross_attention.proj.weight)
                nn.init.zeros_(block.cross_attention.proj.bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        # 输出层置零，模型初始预测的速度场为零。
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def is_conditioning_parameter(name):
        return (
            name == "null_token"
            or name.startswith(("text_norm.", "text_projection."))
            or ".cross_attention." in name
        )

    def conditioning_parameters(self):
        return (
            parameter for name, parameter in self.named_parameters()
            if self.is_conditioning_parameter(name)
        )

    def set_stage(self, stage):
        self.stage = stage
        self.conditional = stage == "text"
        # 无条件阶段冻结文本参数；文本阶段解冻条件分支，同时继续训练主干。
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                self.conditional or not self.is_conditioning_parameter(name)
            )
        return self

    def set_attention(self, implementation):
        for module in self.modules():
            if isinstance(module, (SelfAttention, CrossAttention)):
                module.implementation = implementation
        return self

    def prepare_context(self, context, mask, drop_mask, batch, device):
        if context is None:
            # 空条件仍保留一个有效 token，供文本阶段的无条件预测使用。
            return (
                self.null_token.expand(batch, -1, -1),
                torch.ones(batch, 1, dtype=torch.bool, device=device),
            )
        context = self.text_projection(self.text_norm(context.float()))
        if mask is None:
            mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = mask.to(device=device, dtype=torch.bool)
        if drop_mask is not None:
            null = self.null_token.expand(batch, context.shape[1], -1)
            context = torch.where(drop_mask[:, None, None], null, context)
            # 丢弃整条描述时只开放第一个空 token，避免所有键都被掩掉。
            null_mask = torch.arange(context.shape[1], device=device)[None] == 0
            mask = torch.where(drop_mask[:, None], null_mask, mask)
        return context, mask

    def forward(self, z_t, t, context=None, mask=None, drop_mask=None):
        batch, _, height, width = z_t.shape
        # 每个潜空间位置作为一个 token：[B, C, H, W] 变为 [B, H*W, D]。
        x = self.input_projection(z_t.flatten(2).transpose(1, 2))
        x = x + self.position_embedding.to(x.dtype)
        # 路径时间仍是 [0, 1]，乘 1000 只用于扩展正弦嵌入的相位范围。
        time_embedding = self.time_mlp(sinusoidal_embedding(t * 1000, self.time_dim))
        if self.conditional:
            context, mask = self.prepare_context(context, mask, drop_mask, batch, x.device)
        else:
            context, mask = None, None
        for block in self.blocks:
            x = block(x, time_embedding, context, mask, height, width)
        shift, scale = self.final_modulation(time_embedding).chunk(2, dim=-1)
        x = self.output_projection(modulate(self.final_norm(x), shift, scale))
        # 还原为潜变量网格，输出与输入形状一致的速度预测。
        return x.transpose(1, 2).reshape(batch, self.latent_channels, height, width)
