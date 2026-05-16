"""Helpers shared by train.py and eval.py: seeding and figure generation."""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from channel import transmit_raw


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def save_reconstruction_grid(model, dataset, device, snr_db_list, channel="awgn",
                             n_images=6, out_path="results/reconstructions.png",
                             include_no_coding=True):
    """Visualize originals vs. reconstructions at multiple SNRs.

    If `include_no_coding` is True, each SNR contributes two columns:
      [Raw @ X dB]  — raw pixels sent through the same channel (no encoder/decoder)
      [JSCC @ X dB] — Deep JSCC reconstruction
    This makes the value of learned coding visible at a glance.
    """
    model.eval()
    xs = torch.stack([dataset[i][0] for i in range(n_images)]).to(device)

    cols = [("Original", xs.cpu())]
    for snr in snr_db_list:
        if include_no_coding:
            y_raw = transmit_raw(xs, snr, kind=channel).cpu()
            cols.append((f"Raw @ {snr} dB", y_raw))
        xhat = model(xs, snr, channel=channel).clamp(0, 1).cpu()
        cols.append((f"JSCC @ {snr} dB", xhat))

    n_cols = len(cols)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(1.55 * n_cols, 1.6 * n_images))
    if n_images == 1:
        axes = axes[None, :]
    for j, (title, _) in enumerate(cols):
        axes[0, j].set_title(title, fontsize=9)
    for i in range(n_images):
        for j, (_, imgs) in enumerate(cols):
            axes[i, j].imshow(imgs[i].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
