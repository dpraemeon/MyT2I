import torch
from torch.nn import functional as F


def rf_loss(model, z, context=None, mask=None, drop_mask=None, generator=None):
    z = z.float()
    noise = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=generator)
    t = torch.rand(z.shape[0], device=z.device, generator=generator)
    path_time = t[:, None, None, None]
    # t=0 是噪声，t=1 是图像潜变量，直线路径的目标速度始终是 z - noise。
    z_t = (1 - path_time) * noise + path_time * z
    prediction = model(z_t, t, context=context, mask=mask, drop_mask=drop_mask)
    # 模型前向可以用混合精度，损失仍在 FP32 下计算。
    with torch.autocast(device_type=z.device.type, enabled=False):
        return F.mse_loss(prediction.float(), z - noise)


@torch.no_grad()
def sample(
    model, noise, steps=32, method="euler", context=None, mask=None, cfg_scale=1.0,
):
    z = noise.float().clone()
    dt = 1.0 / steps

    def velocity(state, time):
        t = torch.full((state.shape[0],), time, device=state.device)
        conditional = model(state, t, context=context, mask=mask)
        if context is not None and cfg_scale != 1.0:
            # 无文本前向会走空条件分支，CFG 放大的是有条件与无条件速度之差。
            unconditional = model(state, t)
            conditional = unconditional + cfg_scale * (conditional - unconditional)
        return conditional.float()

    # 与训练路径保持同一方向，从纯噪声沿时间正向积分到图像潜空间。
    for index in range(steps):
        time = index * dt
        first = velocity(z, time)
        if method == "heun":
            # 先用 Euler 预测终点，再取起点和预测终点速度的平均值修正。
            second = velocity(z + dt * first, (index + 1) * dt)
            z = z + 0.5 * dt * (first + second)
        else:
            z = z + dt * first
    return z
