# Deep JSCC — Tutorial Demo

A minimal PyTorch implementation of **Deep Joint Source-Channel Coding (DeepJSCC)** for
wireless image transmission on CIFAR-10. Based on
[Bourtsoulatze et al., 2019](https://ieeexplore.ieee.org/abstract/document/8723589).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samerlahoud/deep-jscc-demo/blob/main/notebooks/demo.ipynb)

## Architecture
![Architecture](architecture.jpg)

Encoder downsamples 32×32×3 to an 8×8×`latent_ch` latent. The latent is power-normalized,
transmitted through a noisy channel (AWGN or Rayleigh), and decoded back to an image.
The whole pipeline is trained end-to-end with MSE.

The bandwidth ratio **k/n = (8·8·latent_ch) / (32·32·3)** — `latent_ch=16` gives
k/n = 1/3, `latent_ch=8` gives 1/6, etc.

## Install
```bash
pip install -r requirements.txt
```

## Quick demo (≈ 1 minute)
Trains a single model on a CIFAR-10 subset and evaluates. Good for a live demo intro.
```bash
python train.py --quick
python eval.py
```

## The two-part tutorial story

### Story 1 — Channel noise and graceful degradation
Train at `SNR_train ∈ {1, 7}` dB, then sweep test SNR. The reconstruction grid puts
raw transmission next to Deep JSCC at the same SNR — the visual elevator pitch.
```bash
python train.py        # fixed: SNR_train ∈ {1, 7} dB (default)
python eval.py         # PSNR/SSIM curves + reconstruction grid
```
Outputs:
- `results/awgn_snr_sweep.png` — PSNR + SSIM vs. SNR_test
- `results/awgn_reconstructions.png` — Original | Raw @ X dB | JSCC @ X dB grid

### Story 2 — Rate-distortion: bandwidth ratio vs. quality
Train at a single SNR across three bandwidth ratios, then plot quality vs. `k/n`.
```bash
python train.py --snr_train_list 7 --latent_ch_list 4 8 16
python eval.py --mode bw_sweep --bw_train_snr 7dB
```
The three `latent_ch` values give `k/n ∈ {1/12, 1/6, 1/3}` — enough to draw the
rate-distortion knee without tripling training time.

Outputs:
- `results/awgn_bw_sweep.png` — PSNR + SSIM vs. bandwidth ratio
- `results/awgn_bw_reconstructions.png` — same image reconstructed at each k/n

## Other variants
```bash
# Rayleigh-fading channel instead of AWGN:
python train.py --channel rayleigh
python eval.py  --channel rayleigh
```

## Outputs (file naming)
Checkpoints encode both the train SNR and the latent size:
```
checkpoints/deepjscc_<channel>_snrtrain_<tag>_lc<N>.pth
                              │              └── latent channels
                              └── e.g. "7dB" or "range_0_20dB"
```

## Reference result (AWGN, paper-style 5-SNR grid)
![AWGN PSNR](awgn_psnr.png)

## Files
| File         | Role                                                                    |
|--------------|-------------------------------------------------------------------------|
| `models.py`  | Encoder / Decoder / `DeepJSCC` wrapper                                  |
| `channel.py` | AWGN + Rayleigh + raw-pixel baseline; `power_normalize`                 |
| `metrics.py` | PSNR + SSIM (pure-torch, no extra deps)                                 |
| `utils.py`   | Seeding + reconstruction-grid visualization                             |
| `train.py`   | Training, both SNR-sweep and bandwidth-sweep modes, `--quick` for demos |
| `eval.py`    | `--mode snr_sweep` or `bw_sweep`; curves + reconstruction grids         |
