"""Deep JSCC encoder / decoder, following Bourtsoulatze et al. (2019), Fig. 2.

For 32x32 CIFAR-10 inputs:
  - Encoder downsamples 32 -> 16 -> 8 with two stride-2 convs.
  - Bottleneck spatial size is 8x8 with `latent_ch` channels.
  - k/n (channel uses per source sample) = (8*8*latent_ch) / (32*32*3).
    latent_ch=8 gives k/n = 1/6 (matching the paper's small-bandwidth point).
"""
import torch.nn as nn

from channel import apply_channel, power_normalize


class Encoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.Conv2d(32, latent_ch, kernel_size=5, stride=1, padding=2),
            nn.PReLU(latent_ch),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(32),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.PReLU(16),
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class DeepJSCC(nn.Module):
    def __init__(self, latent_ch=8):
        super().__init__()
        self.enc = Encoder(latent_ch)
        self.dec = Decoder(latent_ch)

    def forward(self, x, snr_db, channel="awgn"):
        z = self.enc(x)
        z = power_normalize(z)
        y = apply_channel(z, snr_db, kind=channel)
        return self.dec(y)
