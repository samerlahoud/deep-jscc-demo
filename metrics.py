"""Reconstruction-quality metrics for images in [0, 1]."""
import torch
import torch.nn.functional as F


def psnr(x, xhat, eps=1e-8):
    mse = F.mse_loss(xhat, x, reduction="mean").clamp_min(eps)
    return 10.0 * torch.log10(1.0 / mse)


def _gaussian_window(window_size=11, sigma=1.5, channels=3, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(x, xhat, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Mean SSIM over the batch. Inputs in [0, 1], shape (B, C, H, W)."""
    channels = x.size(1)
    window = _gaussian_window(window_size, channels=channels, device=x.device, dtype=x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=channels)
    mu_y = F.conv2d(xhat, window, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(xhat * xhat, window, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * xhat, window, padding=pad, groups=channels) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).mean()
