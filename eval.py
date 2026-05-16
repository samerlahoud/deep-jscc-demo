"""Evaluate Deep JSCC checkpoints across an SNR sweep.

Produces:
  - results/awgn_psnr_curves.png  : PSNR (and SSIM) vs SNR_test, one curve per checkpoint
  - results/reconstructions.png   : qualitative grid of original vs reconstructions
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from metrics import psnr, ssim
from models import DeepJSCC
from utils import save_reconstruction_grid


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
    p.add_argument("--snr_test_list", type=int, nargs="*", default=[1, 7, 19])
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--ckpt_glob", default=None,
                   help="Optional glob to filter checkpoints (default: all matching channel).")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--latent_ch", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  [channel] {args.channel}", flush=True)

    tfm = transforms.ToTensor()
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    pattern = args.ckpt_glob or os.path.join(args.ckpt_dir, f"deepjscc_{args.channel}_*.pth")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}. Run train.py first.")

    os.makedirs(args.out_dir, exist_ok=True)
    psnr_curves, ssim_curves, labels = {}, {}, {}
    models_by_snr = {}  # snr_train -> model, used to pick the grid model

    for ckpt_path in ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        latent_ch = ckpt.get("latent_ch", args.latent_ch)
        model = DeepJSCC(latent_ch=latent_ch).to(device)
        model.load_state_dict(ckpt["state_dict"])

        label = os.path.splitext(os.path.basename(ckpt_path))[0].replace("deepjscc_", "")
        labels[ckpt_path] = label

        p_curve, s_curve = [], []
        for snr_te in args.snr_test_list:
            p, s = eval_psnr_ssim(model, test_loader, device, snr_db=snr_te, channel=args.channel)
            p_curve.append(p)
            s_curve.append(s)
            print(f"  {label:24s} | SNR_test={snr_te:2d}dB | PSNR={p:5.2f} | SSIM={s:.3f}", flush=True)
        psnr_curves[ckpt_path] = p_curve
        ssim_curves[ckpt_path] = s_curve

        snr_train = ckpt.get("snr_train")
        if isinstance(snr_train, (int, float)):
            models_by_snr[float(snr_train)] = model

    # Side-by-side PSNR / SSIM figure.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ckpt_path in ckpts:
        axes[0].plot(args.snr_test_list, psnr_curves[ckpt_path], marker="o", label=labels[ckpt_path])
        axes[1].plot(args.snr_test_list, ssim_curves[ckpt_path], marker="o", label=labels[ckpt_path])
    for ax, ylab in zip(axes, ["PSNR (dB)", "SSIM"]):
        ax.set_xlabel("SNR_test (dB)")
        ax.set_ylabel(ylab)
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_title(f"{args.channel.upper()} channel — PSNR")
    axes[1].set_title(f"{args.channel.upper()} channel — SSIM")
    curves_path = os.path.join(args.out_dir, f"{args.channel}_psnr_ssim_curves.png")
    plt.tight_layout()
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", curves_path)

    # Qualitative reconstructions: pick the model whose SNR_train is closest to
    # the median SNR_test, so the grid shows a well-matched operating point.
    if models_by_snr:
        median_test = sorted(args.snr_test_list)[len(args.snr_test_list) // 2]
        grid_snr = min(models_by_snr, key=lambda s: abs(s - median_test))
        grid_model = models_by_snr[grid_snr]
        print(f"[grid] using model trained at SNR={grid_snr:g} dB", flush=True)
        recon_path = save_reconstruction_grid(
            grid_model, testset, device, args.snr_test_list, channel=args.channel,
            n_images=6, out_path=os.path.join(args.out_dir, f"{args.channel}_reconstructions.png"),
        )
        print("[saved]", recon_path)


if __name__ == "__main__":
    main()
