"""Train Deep JSCC on CIFAR-10.

By default trains one model per SNR in {1, 4, 7, 13, 19} dB (matches the paper).
Use `--snr_train_range LO HI` to train a single model that's robust across an SNR range
(samples a random SNR per batch). `--quick` runs a fast variant for live demos.
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from eval import eval_psnr
from models import DeepJSCC
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--snr_train_list", type=int, nargs="*", default=[1, 4, 7, 13, 19])
    p.add_argument("--snr_train_range", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="If set, train a single model sampling SNR ~ U(LO, HI) per batch.")
    p.add_argument("--channel", choices=["awgn", "rayleigh"], default="awgn")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_ch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="Tiny config for live demos: 1 epoch, single SNR=7 dB, subset of data.")
    return p.parse_args()


def train_one_epoch(model, loader, opt, device, snr_train, channel):
    model.train()
    fixed = not isinstance(snr_train, (tuple, list))
    total, n = 0.0, 0

    pbar = tqdm(loader, leave=False, desc="train")
    for x, _ in pbar:
        x = x.to(device, non_blocking=True)
        snr_db = float(snr_train) if fixed else float(
            torch.empty(1).uniform_(snr_train[0], snr_train[1]).item())

        xhat = model(x, snr_db, channel=channel)
        loss = F.mse_loss(xhat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item() * x.size(0)
        n += x.size(0)
        pbar.set_postfix(mse=f"{total / n:.4f}")
    return total / n


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick:
        args.snr_train_list = [7]
        args.epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  [channel] {args.channel}", flush=True)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    tfm = transforms.ToTensor()
    trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tfm)
    testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)

    if args.quick:
        trainset = torch.utils.data.Subset(trainset, range(2000))
        testset = torch.utils.data.Subset(testset, range(1000))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(testset, batch_size=256, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # Robust single-model mode.
    if args.snr_train_range is not None:
        lo, hi = args.snr_train_range
        tag = f"range_{int(lo)}_{int(hi)}dB"
        configs = [(tag, (lo, hi))]
    else:
        configs = [(f"{s}dB", s) for s in args.snr_train_list]

    for tag, snr_train in configs:
        print(f"\n===== Train @ SNR_train={tag} =====", flush=True)
        model = DeepJSCC(latent_ch=args.latent_ch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            tr = train_one_epoch(model, train_loader, opt, device, snr_train, args.channel)
            eval_snr = snr_train if isinstance(snr_train, (int, float)) else (snr_train[0] + snr_train[1]) / 2
            p = eval_psnr(model, test_loader, device, snr_db=eval_snr, channel=args.channel)
            print(f"Epoch {epoch:02d} | train_mse={tr:.6f} | PSNR@{eval_snr}dB={p:.2f}", flush=True)

        ckpt_path = os.path.join(args.ckpt_dir, f"deepjscc_{args.channel}_snrtrain_{tag}.pth")
        torch.save({
            "snr_train": snr_train,
            "channel": args.channel,
            "latent_ch": args.latent_ch,
            "state_dict": model.state_dict(),
        }, ckpt_path)
        print("[saved]", ckpt_path, flush=True)


if __name__ == "__main__":
    main()
