"""Evaluate Deep JSCC checkpoints.

Two modes:

  - snr_sweep (default): sweeps SNR_test across all checkpoints with matching
    latent_ch. Produces a PSNR/SSIM-vs-SNR plot and a reconstruction grid.

  - bw_sweep: sweeps bandwidth ratio k/n across all checkpoints with matching
    train SNR. Produces a rate-distortion plot.

If a checkpoint trained with --snr_train_range is present, snr_sweep will
include it as a bold "robust" curve.

Outputs go to `results/`.
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from metrics import psnr, ssim
from models import DeepJSCC
from utils import save_reconstruction_grid


# 32x32 CIFAR-10 with two stride-2 conv layers gives an 8x8 latent grid.
LATENT_HW = 8
IMAGE_PIXELS = 32 * 32 * 3


def kn_ratio(latent_ch):
    return (LATENT_HW * LATENT_HW * latent_ch) / IMAGE_PIXELS


@torch.no_grad()
def eval_psnr(model, loader, device, snr_db=10, channel="awgn"):
    model.eval()
    acc, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        xhat = model(x, snr_db, channel=channel).clamp(0, 1)
        acc += psnr(x, xhat).item() * x.size(0)
        n += x.size(0)
    return acc / n


@torch.no_grad()
def eval_psnr_ssim(model, loader, device, snr_db=10, channel="awgn"):
    model.eval()
    p_acc, s_acc, n = 0.0, 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        xhat = model(x, snr_db, channel=channel).clamp(0, 1)
        p_acc += psnr(x, xhat).item() * x.size(0)
        s_acc += ssim(x, xhat).item() * x.size(0)
        n += x.size(0)
    return p_acc / n, s_acc / n


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["snr_sweep", "bw_sweep"], default="snr_sweep")
    p.add_argument("--snr_test_list", type=int, nargs="*", default=[1, 4, 7, 13, 19])
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--latent_ch", type=int, default=16,
                   help="snr_sweep: only use checkpoints with this latent_ch.")
    p.add_argument("--bw_train_snr", default="7dB",
                   help="bw_sweep: load checkpoints whose snrtrain tag matches this.")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def short_label(path):
    """e.g. deepjscc_awgn_snrtrain_7dB_lc16.pth -> snrtrain_7dB_lc16."""
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace("deepjscc_", "").replace(f"{name.split('_')[1]}_", "", 1)


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    latent_ch = ckpt["latent_ch"]
    model = DeepJSCC(latent_ch=latent_ch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def snr_sweep(args, device, testset, test_loader):
    pattern = os.path.join(args.ckpt_dir, f"deepjscc_{args.channel}_snrtrain_*_lc{args.latent_ch}.pth")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}. Run train.py first.")

    print(f"[snr_sweep] {len(ckpts)} checkpoints at latent_ch={args.latent_ch} "
          f"(k/n = {kn_ratio(args.latent_ch):.3f})", flush=True)

    psnr_curves, ssim_curves, labels = {}, {}, {}
    models_by_snr = {}

    for ckpt_path in ckpts:
        model, ckpt = load_ckpt(ckpt_path, device)
        snr_train = ckpt["snr_train"]
        # Label by SNR or range.
        if isinstance(snr_train, (int, float)):
            label = f"fixed {int(snr_train)} dB"
            models_by_snr[float(snr_train)] = model
        else:
            label = f"robust U({int(snr_train[0])}, {int(snr_train[1])}) dB"
        labels[ckpt_path] = label

        p_curve, s_curve = [], []
        for snr_te in args.snr_test_list:
            p, s = eval_psnr_ssim(model, test_loader, device, snr_db=snr_te, channel=args.channel)
            p_curve.append(p)
            s_curve.append(s)
            print(f"  {label:24s} | SNR_test={snr_te:2d}dB | PSNR={p:5.2f} | SSIM={s:.3f}", flush=True)
        psnr_curves[ckpt_path] = p_curve
        ssim_curves[ckpt_path] = s_curve

    # Plot. The robust curve gets a bolder line so it visually stands out as the headline.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ckpt_path in ckpts:
        is_robust = "range" in os.path.basename(ckpt_path)
        kwargs = {"marker": "o", "label": labels[ckpt_path], "linewidth": 2.5 if is_robust else 1.5}
        axes[0].plot(args.snr_test_list, psnr_curves[ckpt_path], **kwargs)
        axes[1].plot(args.snr_test_list, ssim_curves[ckpt_path], **kwargs)
    for ax, ylab in zip(axes, ["PSNR (dB)", "SSIM"]):
        ax.set_xlabel("SNR_test (dB)")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend(fontsize=9)
    axes[0].set_title(f"{args.channel.upper()} — PSNR vs test SNR (k/n = {kn_ratio(args.latent_ch):.3f})")
    axes[1].set_title(f"{args.channel.upper()} — SSIM vs test SNR")
    curves_path = os.path.join(args.out_dir, f"{args.channel}_snr_sweep.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", curves_path)

    # Reconstruction grid from the model closest to the median test SNR.
    if models_by_snr:
        median_test = sorted(args.snr_test_list)[len(args.snr_test_list) // 2]
        grid_snr = min(models_by_snr, key=lambda s: abs(s - median_test))
        grid_model = models_by_snr[grid_snr]
        print(f"[grid] using fixed-SNR model trained at {grid_snr:g} dB", flush=True)
        recon_path = save_reconstruction_grid(
            grid_model, testset, device, args.snr_test_list, channel=args.channel,
            n_images=6, out_path=os.path.join(args.out_dir, f"{args.channel}_reconstructions.png"),
        )
        print("[saved]", recon_path)


def bw_sweep(args, device, testset, test_loader):
    pattern = os.path.join(
        args.ckpt_dir, f"deepjscc_{args.channel}_snrtrain_{args.bw_train_snr}_lc*.pth")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}. "
                                f"Try: python train.py --snr_train_list 7 --latent_ch_list 4 8 16 24")

    print(f"[bw_sweep] {len(ckpts)} checkpoints at snrtrain={args.bw_train_snr}", flush=True)

    # latent_ch -> (model, k/n)
    entries = []
    for ckpt_path in ckpts:
        model, ckpt = load_ckpt(ckpt_path, device)
        lc = ckpt["latent_ch"]
        entries.append((lc, kn_ratio(lc), model))
    entries.sort(key=lambda e: e[0])

    # PSNR matrix: rows = test SNR, cols = bandwidth ratio.
    psnr_by_snr = {snr: [] for snr in args.snr_test_list}
    ssim_by_snr = {snr: [] for snr in args.snr_test_list}
    for lc, kn, model in entries:
        for snr_te in args.snr_test_list:
            p, s = eval_psnr_ssim(model, test_loader, device, snr_db=snr_te, channel=args.channel)
            psnr_by_snr[snr_te].append(p)
            ssim_by_snr[snr_te].append(s)
            print(f"  latent_ch={lc:2d} (k/n={kn:.3f}) | SNR_test={snr_te:2d}dB | "
                  f"PSNR={p:5.2f} | SSIM={s:.3f}", flush=True)

    xs = [kn for _, kn, _ in entries]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for snr in args.snr_test_list:
        axes[0].plot(xs, psnr_by_snr[snr], marker="o", label=f"SNR_test = {snr} dB")
        axes[1].plot(xs, ssim_by_snr[snr], marker="o", label=f"SNR_test = {snr} dB")
    for ax, ylab in zip(axes, ["PSNR (dB)", "SSIM"]):
        ax.set_xlabel("Bandwidth ratio k/n")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend(fontsize=9)
    axes[0].set_title(f"{args.channel.upper()} — rate-distortion (train SNR = {args.bw_train_snr})")
    axes[1].set_title(f"{args.channel.upper()} — SSIM vs k/n")
    curves_path = os.path.join(args.out_dir, f"{args.channel}_bw_sweep.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", curves_path)

    # Show the same image reconstructed at each bandwidth, fixed at the middle test SNR.
    mid_snr = sorted(args.snr_test_list)[len(args.snr_test_list) // 2]
    n_images = 6
    xs_data = torch.stack([testset[i][0] for i in range(n_images)]).to(device)
    cols = [("Original", xs_data.cpu())]
    for lc, kn, model in entries:
        with torch.no_grad():
            xhat = model(xs_data, mid_snr, channel=args.channel).clamp(0, 1).cpu()
        cols.append((f"k/n={kn:.3f}\n(lc={lc})", xhat))

    n_cols = len(cols)
    fig, axes = plt.subplots(n_images, n_cols, figsize=(1.55 * n_cols, 1.7 * n_images))
    if n_images == 1:
        axes = axes[None, :]
    for j, (title, _) in enumerate(cols):
        axes[0, j].set_title(title, fontsize=9)
    for i in range(n_images):
        for j, (_, imgs) in enumerate(cols):
            axes[i, j].imshow(imgs[i].permute(1, 2, 0).numpy())
            axes[i, j].axis("off")
    fig.suptitle(f"Reconstructions at SNR_test = {mid_snr} dB across bandwidth ratios", fontsize=11)
    bw_grid_path = os.path.join(args.out_dir, f"{args.channel}_bw_reconstructions.png")
    plt.tight_layout()
    plt.savefig(bw_grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", bw_grid_path)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  [channel] {args.channel}  [mode] {args.mode}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tfm = transforms.ToTensor()
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    if args.mode == "snr_sweep":
        snr_sweep(args, device, testset, test_loader)
    else:
        bw_sweep(args, device, testset, test_loader)


if __name__ == "__main__":
    main()
