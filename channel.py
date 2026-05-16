"""Channel models for Deep JSCC.

`power_normalize` enforces unit average power per sample on the encoder output,
so that `SNR (dB)` directly controls the noise variance.

Two channels are provided:
  - awgn:     y = x + n,                n ~ N(0, 1/SNR)
  - rayleigh: y = h * x + n,             h ~ CN(0, 1) (real magnitude here), n ~ N(0, 1/SNR)
"""
import torch


def power_normalize(x, eps=1e-8):
    p = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return x / torch.sqrt(p)


def awgn(x, snr_db):
    snr_lin = 10 ** (snr_db / 10)
    noise_std = (1.0 / snr_lin) ** 0.5
    return x + torch.randn_like(x) * noise_std


def rayleigh(x, snr_db):
    # Real-valued surrogate: |h| ~ Rayleigh(1), one fade coefficient per sample.
    snr_lin = 10 ** (snr_db / 10)
    noise_std = (1.0 / snr_lin) ** 0.5
    B = x.size(0)
    hr = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype)
    hi = torch.randn(B, 1, 1, 1, device=x.device, dtype=x.dtype)
    h = torch.sqrt(0.5 * (hr * hr + hi * hi))
    return h * x + torch.randn_like(x) * noise_std


def apply_channel(x, snr_db, kind="awgn"):
    if kind == "awgn":
        return awgn(x, snr_db)
    if kind == "rayleigh":
        return rayleigh(x, snr_db)
    raise ValueError(f"unknown channel: {kind}")
