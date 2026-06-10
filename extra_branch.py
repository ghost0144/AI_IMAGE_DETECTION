import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):

    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(32, out_dim)

    def forward(self, x):

        # FFT
        fft = torch.fft.fft2(x)
        fft = torch.abs(fft)

        # log幅度
        fft = torch.log(fft + 1e-6)

        feat = self.conv(fft)
        feat = feat.view(feat.size(0), -1)

        feat = self.fc(feat)

        return feat