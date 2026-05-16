"""Helpers shared by train.py and eval.py: seeding and figure generation."""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def save_reconstruction_grid(model, dataset, device, snr_db_list, channel="awgn",
                             n_images=6, out_path="results/reconstructions.png"):
    """Visualize: each row is a sample image, columns are original + reconstructions at given SNRs."""
    model.eval()
    xs = torch.stack([dataset[i][0] for i in range(n_images)]).to(device)

    cols = [xs.cpu()]
    for snr in snr_db_list:
        xhat = model(xs, snr, channel=channel).clamp(0, 1).cpu()
        cols.append(xhat)

    fig, axes = plt.subplots(n_images, len(cols), figsize=(1.6 * len(cols), 1.6 * n_images))
    if n_images == 1:
        axes = axes[None, :]
    titles = ["Original"] + [f"{s} dB" for s in snr_db_list]
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=10)
    for i in range(n_images):
        for j in range(len(cols)):
            axes[i, j].imshow(cols[j][i].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
