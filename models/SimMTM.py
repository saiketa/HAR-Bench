import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_sequence_encoder


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0.0
        for i, loss in enumerate(losses):
            loss_sum = loss_sum + 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class ContrastiveWeight(nn.Module):
    def __init__(self, temperature=0.2, positive_nums=3):
        super().__init__()
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.positive_nums = positive_nums

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size):
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag).to(similarity_matrix.device).bool()

        oral_batch_size = cur_batch_size // (self.positive_nums + 1)
        positives_mask = np.zeros(similarity_matrix.size(), dtype=np.float32)
        for i in range(self.positive_nums + 1):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0
        return positives_mask.bool(), negatives_mask.bool()

    def forward(self, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        positives_mask, negatives_mask = self.get_positive_and_negative_mask(
            similarity_matrix, cur_batch_shape[0]
        )
        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (
                torch.ones(cur_batch_shape[0], positives.shape[-1], device=batch_emb_om.device),
                torch.zeros(cur_batch_shape[0], negatives.shape[-1], device=batch_emb_om.device),
            ),
            dim=-1,
        ).float()

        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)
        return loss, similarity_matrix, logits, positives_mask


class AggregationRebuild(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape
        similarity_matrix = similarity_matrix / self.temperature
        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0], device=similarity_matrix.device) * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)
        return rebuild_weight_matrix, rebuild_oral_batch_emb


def geom_noise_mask_single(length, lm, masking_ratio):
    keep_mask = np.ones(length, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]

    state = int(np.random.rand() > masking_ratio)
    for i in range(length):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask


def noise_mask(x, masking_ratio=0.25, lm=3, distribution="geometric"):
    if distribution == "geometric":
        mask = geom_noise_mask_single(x.shape[0] * x.shape[1] * x.shape[2], lm, masking_ratio)
        mask = mask.reshape(x.shape[0], x.shape[1], x.shape[2])
    else:
        mask = np.random.choice(
            np.array([True, False]),
            size=x.shape,
            replace=True,
            p=(1 - masking_ratio, masking_ratio),
        )
    return torch.tensor(mask, device=x.device)


def data_transform_masked4cl(sample, masking_ratio, lm, positive_nums=None, distribution="geometric"):
    if positive_nums is None:
        positive_nums = math.ceil(1.5 / (1 - masking_ratio))

    sample = sample.permute(0, 2, 1)
    sample_repeat = sample.repeat(positive_nums, 1, 1)

    mask = noise_mask(sample_repeat, masking_ratio, lm, distribution=distribution)
    x_masked = mask * sample_repeat
    return x_masked.permute(0, 2, 1), mask.permute(0, 2, 1)


class SimMTMBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = build_sequence_encoder(cfg)
        proj_dim = int(getattr(cfg, "proj_dim", 128) or 128)
        flat_dim = int(cfg.seq_len) * int(cfg.hidden)

        self.dense = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        self.rebuild_head = nn.Linear(flat_dim, int(cfg.seq_len) * int(cfg.feature_num))

    def encode_sequence(self, x):
        return self.encoder(x)

    def encode_projected(self, x):
        seq_features = self.encode_sequence(x)
        flat_features = seq_features.reshape(seq_features.size(0), -1)
        proj_features = self.dense(flat_features)
        return seq_features, flat_features, proj_features


class SimMTMModel4Pretrain(nn.Module):
    def __init__(self, cfg, temperature=0.2, positive_nums=3):
        super().__init__()
        self.cfg = cfg
        self.backbone = SimMTMBackbone(cfg)
        self.awl = AutomaticWeightedLoss(2)
        self.contrastive = ContrastiveWeight(temperature=temperature, positive_nums=positive_nums)
        self.aggregation = AggregationRebuild(temperature=temperature)
        self.mse = nn.MSELoss()

    def configure_objectives(self, temperature=0.2, positive_nums=3):
        self.contrastive = ContrastiveWeight(temperature=temperature, positive_nums=positive_nums)
        self.aggregation = AggregationRebuild(temperature=temperature)

    def forward(self, x_in_t, pretrain=False):
        seq_features, flat_features, proj_features = self.backbone.encode_projected(x_in_t)

        if not pretrain:
            return flat_features, proj_features

        loss_cl, similarity_matrix, _, _ = self.contrastive(proj_features)
        _, agg_x = self.aggregation(similarity_matrix, seq_features)
        pred_x = self.backbone.rebuild_head(agg_x.reshape(agg_x.size(0), -1))
        loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
        loss = self.awl(loss_cl, loss_rb)

        return {
            "loss": loss,
            "loss_cl": loss_cl,
            "loss_rb": loss_rb,
            "sequence_features": seq_features,
            "flat_features": flat_features,
            "proj_features": proj_features,
        }

    def backbone_features(self, x):
        return self.backbone.encode_sequence(x)
