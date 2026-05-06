import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        return F.gelu(h + residual)


class ResNet1DEncoder(nn.Module):
    """ResNet-style temporal encoder with sequence-length-preserving residual blocks.

    Input: [B, T, C]
    Output: [B, T, H]
    """

    def __init__(self, cfg):
        super().__init__()
        feature_num = int(getattr(cfg, "feature_num", 6))
        hidden = int(getattr(cfg, "hidden", 72))
        kernel_size = int(getattr(cfg, "resnet_kernel_size", 5))
        dropout = float(getattr(cfg, "resnet_dropout", 0.1))
        num_blocks = int(getattr(cfg, "resnet_blocks", 4))
        dilations = list(getattr(cfg, "resnet_dilations", [1, 2, 4, 8]))
        if len(dilations) == 0:
            dilations = [1]

        self.stem = nn.Sequential(
            nn.Conv1d(feature_num, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(
                    channels=hidden,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        self.out_norm = nn.GroupNorm(1, hidden)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"ResNet1DEncoder expects [B, T, C], got {tuple(x.shape)}")
        h = x.transpose(1, 2)  # [B, C, T]
        h = self.stem(h)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        return h.transpose(1, 2).contiguous()


class TemporalConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)
        return h


class CNN1DEncoder(nn.Module):
    """Plain temporal CNN encoder with dilation growth and fixed output shape.

    Input: [B, T, C]
    Output: [B, T, H]
    """

    def __init__(self, cfg):
        super().__init__()
        feature_num = int(getattr(cfg, "feature_num", 6))
        hidden = int(getattr(cfg, "hidden", 72))
        kernel_size = int(getattr(cfg, "cnn_kernel_size", 5))
        dropout = float(getattr(cfg, "cnn_dropout", 0.1))
        num_blocks = int(getattr(cfg, "cnn_depth", 4))
        dilations = list(getattr(cfg, "cnn_dilations", [1, 2, 4, 8]))
        expansion = int(getattr(cfg, "cnn_hidden_multiplier", 2))
        if len(dilations) == 0:
            dilations = [1]
        mid_channels = max(hidden, hidden * expansion)

        blocks = []
        in_channels = feature_num
        for i in range(num_blocks):
            out_channels = hidden if i == num_blocks - 1 else mid_channels
            blocks.append(
                TemporalConvBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.out_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.out_norm = nn.GroupNorm(1, hidden)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"CNN1DEncoder expects [B, T, C], got {tuple(x.shape)}")
        h = x.transpose(1, 2)  # [B, C, T]
        h = self.network(h)
        h = self.out_proj(h)
        h = self.out_norm(h)
        return h.transpose(1, 2).contiguous()


def build_sequence_encoder(cfg):
    encoder_type = str(getattr(cfg, "encoder_type", "transformer")).lower()
    if encoder_type == "transformer":
        from .LIMU_BERT import Transformer
        return Transformer(cfg)
    if encoder_type == "resnet":
        return ResNet1DEncoder(cfg)
    if encoder_type == "cnn":
        return CNN1DEncoder(cfg)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")
