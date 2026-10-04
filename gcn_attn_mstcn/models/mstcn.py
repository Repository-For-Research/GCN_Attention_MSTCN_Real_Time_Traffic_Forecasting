import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]


class MSTCNBlock(nn.Module):
    """Three parallel dilated causal-conv branches (k = 3, 5, 7), fused by a 1x1 conv."""

    DILATION_PLANS = ([1, 2, 4, 8], [2, 2, 4, 8], [8, 4, 2, 1])
    KERNEL_SIZES = (3, 5, 7)

    def __init__(self, channels: int, dropout: float = 0.2):
        super().__init__()
        out_ch = channels // 3
        self.branches = nn.ModuleList()
        for k, dilations in zip(self.KERNEL_SIZES, self.DILATION_PLANS):
            layers, in_ch = [], channels
            for d in dilations:
                layers += [CausalConv1d(in_ch, out_ch, k, d), nn.ReLU(), nn.Dropout(dropout)]
                in_ch = out_ch
            self.branches.append(nn.Sequential(*layers))
        self.fuse = nn.Conv1d(out_ch * 3, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        out = self.fuse(torch.cat([b(x) for b in self.branches], dim=1))
        return self.norm(out) + x
