# modules.py (fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostModule(nn.Module):
    """
    Robust GhostModule: handles small channel counts safely.
    Produces 'oup' output channels. If new_channels == 0, the cheap path is identity.
    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=False):
        super().__init__()
        self.oup = oup
        # ensure at least 1 channel in the primary path
        init_channels = max(1, int(oup / ratio))
        # remaining channels produced by cheap operation
        new_channels = max(0, oup - init_channels)

        # primary conv produces init_channels
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Identity(),
        )

        # cheap operation: depthwise conv that produces new_channels
        if new_channels > 0:
            # groups must equal input channels for depthwise conv; since input to cheap_op is init_channels,
            # valid groups == init_channels (init_channels >= 1 guaranteed).
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size//2,
                          groups=init_channels, bias=False),
                nn.BatchNorm2d(new_channels),
                nn.SiLU(inplace=True) if relu else nn.Identity(),
            )
        else:
            # no cheap op needed; use identity mapping that returns zero-tensor slice later
            self.cheap_operation = None

    def forward(self, x):
        x1 = self.primary_conv(x)
        if self.cheap_operation is not None:
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1
        # Ensure exact 'oup' channels (slice if too large)
        if out.shape[1] >= self.oup:
            out = out[:, :self.oup, :, :].contiguous()
        else:
            # pad with zeros if necessary (rare)
            pad_ch = self.oup - out.shape[1]
            pad = torch.zeros((out.shape[0], pad_ch, out.shape[2], out.shape[3]), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=1)
        return out

class GhostConvBlock(nn.Module):
    """
    GhostConvBlock: replacement for Conv+BN+SiLU blocks.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.ghost = GhostModule(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.ghost(x))

class SimAM(nn.Module):
    """
    SimAM attention (parameter-free).
    Lightweight attention that computes per-element saliency.
    """
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        # flatten spatial dims
        x_flat = x.view(b, c, -1)  # (B, C, H*W)
        mean = x_flat.mean(-1, keepdim=True)  # (B, C, 1)
        var = x_flat.var(-1, keepdim=True, unbiased=False)  # (B, C, 1)
        numerator = (x_flat - mean)**2
        denom = 4 * (var + self.e_lambda)
        energy = (numerator / (denom + 1e-12)) + 0.5
        attention = torch.sigmoid(energy).view(b, c, h, w)
        return x * attention
