import torch
import torch.nn as nn

from .encoders import build_sequence_encoder


def split_features(mod_features):
    """Split each modality feature into shared/private halves (same as original FOCAL)."""
    split_mod_features = {}

    for mod in mod_features:
        if mod_features[mod].ndim == 2:
            split_dim = mod_features[mod].shape[1] // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, 0:split_dim],
                "private": mod_features[mod][:, split_dim:],
            }
        else:
            b, seq, dim = mod_features[mod].shape
            split_dim = dim // 2
            split_mod_features[mod] = {
                "shared": mod_features[mod][:, :, 0:split_dim],
                "private": mod_features[mod][:, :, split_dim : 2 * split_dim],
            }

    return split_mod_features


def extract_non_diagonal_matrix(input_tensor):
    """Extract non-diagonal elements at the last two dims.

    input shape: [b, n, n]
    output shape: [b, n, n-1]
    """
    flatten_input = input_tensor.reshape([-1, input_tensor.shape[-2], input_tensor.shape[-1]])
    b, n, _ = flatten_input.shape

    non_diagonal_input = flatten_input.flatten(start_dim=1)[:, 1:]
    non_diagonal_input = non_diagonal_input.view(b, n - 1, n + 1)[:, :, :-1]
    non_diagonal_input = non_diagonal_input.reshape([b, n, n - 1])

    return non_diagonal_input


def adapt_focal_checkpoint(state_dict, use_dual_modalities=True):
    """Adapt legacy single-projector FOCAL checkpoints to the dual-modality layout."""
    if not use_dual_modalities:
        return state_dict

    has_dual = any(k.startswith("backbone.mod_projectors.") for k in state_dict.keys())
    has_single = any(k.startswith("backbone.mod_projector.") for k in state_dict.keys())

    if has_dual or not has_single:
        return state_dict

    adapted = dict(state_dict)
    prefix = "backbone.mod_projector."
    for key, value in list(state_dict.items()):
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            adapted[f"backbone.mod_projectors.acc.{suffix}"] = value
            adapted[f"backbone.mod_projectors.gyro.{suffix}"] = value
            adapted.pop(key, None)

    return adapted


class FOCALBackbone(nn.Module):
    """FOCAL backbone adapted to HAR-Bench encoder convention.

    Encoder is fixed to LIMU_BERT Transformer as requested.
    """

    def __init__(self, cfg):
        super().__init__()
        self.raw_feature_num = int(getattr(cfg, "feature_num", 6))
        self.feature_num = self.raw_feature_num * 2
        encoder_cfg = cfg._replace(feature_num=self.feature_num) if hasattr(cfg, "_replace") else cfg
        if not hasattr(cfg, "_replace"):
            encoder_cfg.feature_num = self.feature_num
        self.transformer = build_sequence_encoder(encoder_cfg)
        self.use_dual_modalities = bool(getattr(cfg, "focal_use_dual_modalities", True)) and self.raw_feature_num >= 6

        proj_dim = getattr(cfg, "hidden", 128)
        if proj_dim % 2 != 0:
            raise ValueError(f"FOCAL projection dim must be even for shared/private split, got {proj_dim}")

        if self.use_dual_modalities:
            self.mod_projectors = nn.ModuleDict({
                "acc": nn.Sequential(
                    nn.Linear(cfg.hidden, proj_dim),
                    nn.ReLU(),
                    nn.Linear(proj_dim, proj_dim),
                ),
                "gyro": nn.Sequential(
                    nn.Linear(cfg.hidden, proj_dim),
                    nn.ReLU(),
                    nn.Linear(proj_dim, proj_dim),
                ),
            })
        else:
            self.mod_projector = nn.Sequential(
                nn.Linear(cfg.hidden, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

    def _maybe_fft_input(self, x):
        if x.size(-1) == self.feature_num:
            return x
        if x.size(-1) != self.raw_feature_num:
            raise ValueError(
                f"FOCAL expected raw {self.raw_feature_num} channels or FFT {self.feature_num} channels, "
                f"got {x.shape}"
            )
        z = torch.fft.fft(x, dim=1)
        return torch.stack((z.real, z.imag), dim=-1).reshape(x.size(0), x.size(1), x.size(2) * 2)

    def forward(self, x, proj_head=False):
        x = self._maybe_fft_input(x)

        if not self.use_dual_modalities:
            h = self.transformer(x)  # [B, T, H]
            h = h.mean(dim=1)        # [B, H], align with original FOCAL sample-level feature
            if proj_head:
                h = self.mod_projector(h)
            return {"imu": h}

        # Dual-modality adaptation for IMU:
        # FOCAL preprocess follows the official FFT packing, so each raw
        # channel becomes adjacent real/imag channels.
        if x.size(-1) < 6:
            raise ValueError(f"FOCAL dual-modality expects input with >=6 channels, got {x.shape}")
        mod_width = x.size(-1) // 2

        x_acc = x.clone()
        x_acc[:, :, mod_width:] = 0.0
        x_gyro = x.clone()
        x_gyro[:, :, :mod_width] = 0.0

        h_acc = self.transformer(x_acc).mean(dim=1)    # [B, H]
        h_gyro = self.transformer(x_gyro).mean(dim=1)  # [B, H]

        if proj_head:
            h_acc = self.mod_projectors["acc"](h_acc)
            h_gyro = self.mod_projectors["gyro"](h_gyro)

        return {"acc": h_acc, "gyro": h_gyro}

    def backbone_features(self, x):
        x = self._maybe_fft_input(x)
        return self.transformer(x)


class FOCAL4Pretrain(nn.Module):
    """FOCAL pretraining model with HAR-Bench unified backbone interface."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = FOCALBackbone(cfg)

    def forward(self, aug_input1, aug_input2=None, proj_head=True):
        if aug_input2 is None:
            return self.backbone_features(aug_input1)

        mod_features1 = self.backbone(aug_input1, proj_head=proj_head)
        mod_features2 = self.backbone(aug_input2, proj_head=proj_head)
        return mod_features1, mod_features2

    def backbone_features(self, x):
        return self.backbone.backbone_features(x)


class FOCALLoss(nn.Module):
    """Original FOCAL objective with single-modality HAR adaptation."""

    def __init__(
        self,
        device,
        seq_len,
        modalities=None,
        temperature=0.07,
        inter_rank_margin=1.0,
        shared_contrastive_loss_weight=1.0,
        private_contrastive_loss_weight=1.0,
        orthogonal_loss_weight=3.0,
        rank_loss_weight=5.0,
        no_private=False,
    ):
        super().__init__()
        self.device = device
        self.seq_len = int(seq_len)
        self.modalities = ["imu"] if modalities is None else list(modalities)
        self.temperature = float(temperature)
        self.no_private = bool(no_private)

        self.shared_contrastive_loss_weight = float(shared_contrastive_loss_weight)
        self.private_contrastive_loss_weight = float(private_contrastive_loss_weight)
        self.orthogonal_loss_weight = float(orthogonal_loss_weight)
        self.rank_loss_weight = float(rank_loss_weight)

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.orthogonal_loss_f = nn.CosineEmbeddingLoss(reduction="mean")
        self.inter_ranking_loss_f = nn.MarginRankingLoss(margin=float(inter_rank_margin), reduction="mean")

    def _reshape_to_bsd(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Convert feature to [B, S, D] for FOCAL loss.

        Accept:
            [B, S, D] -> unchanged
            [N, D]    -> try reshape with configured seq_len; if not divisible, fallback to [N, 1, D]
        """
        if feat.ndim == 3:
            return feat

        if feat.ndim == 2:
            n, d = feat.shape
            if self.seq_len > 1 and n % self.seq_len == 0:
                return feat.reshape(n // self.seq_len, self.seq_len, d)
            return feat.unsqueeze(1)

        raise ValueError(f"Unsupported feature shape for FOCAL loss: {tuple(feat.shape)}")

    def mask_correlated_samples(self, seq_len, batch_size, temporal=False):
        if temporal:
            mask = torch.ones([batch_size, batch_size], dtype=bool, device=self.device)
            mask = mask.fill_diagonal_(False)
            mask = mask.repeat_interleave(seq_len, dim=0).repeat_interleave(seq_len, dim=1)
        else:
            n = 2 * batch_size
            diag_mat = torch.eye(batch_size, device=self.device)
            mask = torch.ones((n, n), device=self.device)

            mask = mask.fill_diagonal_(0)
            mask[0:batch_size, batch_size : 2 * batch_size] -= diag_mat
            mask[batch_size : 2 * batch_size, 0:batch_size] -= diag_mat

            mask = mask.unsqueeze(0).repeat(seq_len, 1, 1).bool()

        return mask

    def forward_contrastive_loss(self, embeddings1, embeddings2, finegrain=False):
        batch, seq, _ = embeddings1.shape

        if finegrain:
            in_embeddings1 = embeddings1
            in_embeddings2 = embeddings2
            n = 2 * seq
            dim_parallel = batch
            dim_compare = seq
        else:
            in_embeddings1 = embeddings1.transpose(0, 1)
            in_embeddings2 = embeddings2.transpose(0, 1)
            n = 2 * batch
            dim_parallel = seq
            dim_compare = batch

        z = torch.cat((in_embeddings1, in_embeddings2), dim=1)
        sim = self.similarity_f(z.unsqueeze(2), z.unsqueeze(1)) / self.temperature
        sim_i_j = torch.diagonal(sim, dim_compare, dim1=-2, dim2=-1)
        sim_j_i = torch.diagonal(sim, -dim_compare, dim1=-2, dim2=-1)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=1).reshape(dim_parallel, n, 1)
        negative_samples = sim[self.mask_correlated_samples(dim_parallel, dim_compare)].reshape(dim_parallel, n, -1)

        labels = torch.zeros(dim_parallel * n, device=positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=2).reshape(dim_parallel * n, -1)
        contrastive_loss = self.criterion(logits, labels)

        return contrastive_loss

    def forward_orthogonality_loss(self, embeddings1, embeddings2):
        flat_embeddings1 = embeddings1.reshape(-1, embeddings1.shape[-1])
        flat_embeddings2 = embeddings2.reshape(-1, embeddings2.shape[-1])

        batch = flat_embeddings1.shape[0]
        orthogonal_loss = self.orthogonal_loss_f(
            flat_embeddings1,
            flat_embeddings2,
            target=-torch.ones(batch, device=embeddings1.device),
        )
        return orthogonal_loss

    def forward_temporal_inter_ranking_loss(self, embeddings):
        batch_size, seq_len, dim = embeddings.shape
        if batch_size <= 1:
            return embeddings.new_tensor(0.0)
        in_embeddings = embeddings.reshape(batch_size * seq_len, dim)

        distance = torch.cdist(in_embeddings, in_embeddings, p=2)
        distance = distance.reshape(batch_size, seq_len, batch_size, seq_len)
        distance = distance.permute(0, 2, 1, 3)

        mask = torch.ones(batch_size * seq_len, batch_size * seq_len, device=self.device).fill_diagonal_(0)
        mask = mask.reshape(batch_size, seq_len, batch_size, seq_len).permute(0, 2, 1, 3)
        distance = (distance * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp_min(1.0)

        avg_intra_seq_dist = torch.diagonal(distance, 0, dim1=0, dim2=1).repeat_interleave(batch_size - 1)
        avg_inter_seq_dist = extract_non_diagonal_matrix(distance).flatten()

        ranking_loss = self.inter_ranking_loss_f(
            avg_intra_seq_dist,
            avg_inter_seq_dist,
            -torch.ones_like(avg_intra_seq_dist, device=self.device),
        )
        return ranking_loss

    def forward(self, mod_features1, mod_features2, index=None):
        reshaped_mod_features1, reshaped_mod_features2 = {}, {}
        for mod in self.modalities:
            reshaped_mod_features1[mod] = self._reshape_to_bsd(mod_features1[mod])
            reshaped_mod_features2[mod] = self._reshape_to_bsd(mod_features2[mod])

        split_mod_features1 = split_features(reshaped_mod_features1)
        split_mod_features2 = split_features(reshaped_mod_features2)

        shared_contrastive_loss = 0.0
        if self.no_private:
            for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            mod_features[mod1], mod_features[mod2]
                        )
        else:
            for split_mod_features in [split_mod_features1, split_mod_features2]:
                for i, mod1 in enumerate(self.modalities):
                    for mod2 in self.modalities[i + 1 :]:
                        shared_contrastive_loss += self.forward_contrastive_loss(
                            split_mod_features[mod1]["shared"], split_mod_features[mod2]["shared"]
                        )

        private_contrastive_loss = 0.0
        for mod in self.modalities:
            private_contrastive_loss += self.forward_contrastive_loss(
                split_mod_features1[mod]["private"],
                split_mod_features2[mod]["private"],
            )

        temporal_consistency_loss = 0.0
        for mod_features in [reshaped_mod_features1, reshaped_mod_features2]:
            for mod in self.modalities:
                temporal_consistency_loss += self.forward_temporal_inter_ranking_loss(mod_features[mod])

        orthogonality_loss = 0.0
        for split_mod_features in [split_mod_features1, split_mod_features2]:
            for i, mod in enumerate(self.modalities):
                orthogonality_loss += self.forward_orthogonality_loss(
                    split_mod_features[mod]["shared"], split_mod_features[mod]["private"]
                )

                for mod2 in self.modalities[i + 1 :]:
                    orthogonality_loss += self.forward_orthogonality_loss(
                        split_mod_features[mod]["private"], split_mod_features[mod2]["private"]
                    )

        loss = (
            shared_contrastive_loss * self.shared_contrastive_loss_weight
            + private_contrastive_loss * self.private_contrastive_loss_weight
            + orthogonality_loss * self.orthogonal_loss_weight
            + temporal_consistency_loss * self.rank_loss_weight
        )
        return loss
