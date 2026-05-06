import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoders import build_sequence_encoder


def generate_continuous_mask(batch_size, seq_len, n=5, length=0.1):
    mask = torch.full((batch_size, seq_len), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * seq_len)
    n = max(min(n, seq_len // 2), 1)

    if isinstance(length, float):
        length = int(length * seq_len)
    length = max(length, 1)

    for i in range(batch_size):
        for _ in range(n):
            t = np.random.randint(seq_len - length + 1)
            mask[i, t:t + length] = False
    return mask


def generate_binomial_mask(batch_size, seq_len, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(batch_size, seq_len))).to(torch.bool)


def instance_contrastive_loss(z1, z2):
    bsz, ts_len = z1.size(0), z1.size(1)
    if bsz == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=0)  # [2B, T, C]
    z = z.transpose(0, 1)  # [T, 2B, C]
    sim = torch.matmul(z, z.transpose(1, 2))  # [T, 2B, 2B]
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(bsz, device=z1.device)
    loss = (logits[:, i, bsz + i - 1].mean() + logits[:, bsz + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    bsz, ts_len = z1.size(0), z1.size(1)
    if ts_len == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=1)  # [B, 2T, C]
    sim = torch.matmul(z, z.transpose(1, 2))  # [B, 2T, 2T]
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(ts_len, device=z1.device)
    loss = (logits[:, t, ts_len + t - 1].mean() + logits[:, ts_len + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    depth = 0

    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        if depth >= temporal_unit and (1 - alpha) != 0:
            loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)

        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        depth += 1

    return loss / max(depth, 1)


class TS2VecEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = build_sequence_encoder(cfg)
        self.repr_dropout = nn.Dropout(p=0.1)
        self.mask_mode = getattr(cfg, "ts2vec_mask_mode", "binomial")

    def forward(self, x, mask=None):
        nan_mask = ~x.isnan().any(dim=-1)
        x = x.masked_fill(~nan_mask.unsqueeze(-1), 0.0)

        if mask is None:
            mask = self.mask_mode if self.training else "all_true"

        if mask == "binomial":
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask = mask & nan_mask
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return self.repr_dropout(self.transformer(x))


class TS2VecModel4Pretrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._net = TS2VecEncoder(cfg)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

    def forward(self, x):
        return self._net(x)

    def backbone_features(self, x):
        return self.net(x)
