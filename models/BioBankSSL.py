import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_sequence_encoder


class BioBankSSL4Pretrain(nn.Module):
    """
    Multi-task self-supervised pretraining model for BioBankSSL.

    Tasks:
        1. time reversal
        2. permutation
        3. time warp

    Input:
        x: [B, T, feature_num]

    Output:
        aot_pred:      [B, 2]
        permute_pred:  [B, 2]
        time_w_pred:   [B, 2]
        features:      [B, H, T]
    """
    def __init__(self, cfg, num_classes=2):
        super().__init__()
        self.transformer = build_sequence_encoder(cfg)

        self.hidden_dim = cfg.hidden
        self.num_classes = num_classes

        # Three task heads
        self.aot_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.permute_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.time_w_head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, T, feature_num]

        Returns:
            aot_pred: [B, 2]
            permute_pred: [B, 2]
            time_w_pred: [B, 2]
            features: [B, H, T]
        """
        # Backbone output: [B, T, H]
        h = self.transformer(x)

        # Global temporal pooling
        pooled = h.mean(dim=1)                  # [B, H]

        # Multi-task predictions
        aot_pred = self.aot_head(pooled)        # [B, 2]
        permute_pred = self.permute_head(pooled) # [B, 2]
        time_w_pred = self.time_w_head(pooled)   # [B, 2]

        # for optional downstream usage
        features = h.transpose(1, 2)            # [B, H, T]

        return aot_pred, permute_pred, time_w_pred, features
