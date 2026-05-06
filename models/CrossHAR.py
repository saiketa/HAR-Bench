import torch
import torch.nn as nn

from .encoders import build_sequence_encoder
from .attention import Seq_Transformer


class MaskedModel4Pretrain(nn.Module):
    """
    CrossHAR masked pretraining model.

    Input:
        input_seqs: [B, T, feature_num]
        masked_pos: [B, N_mask] or None

    Output:
        representation: [B, T, hidden]
        logits_lm:      [B, N_mask, feature_num] if masked_pos is not None
                        [B, T, feature_num] otherwise
    """
    def __init__(self, cfg, output_embed=False):
        super().__init__()
        self.transformer = build_sequence_encoder(cfg)
        self.output_embed = output_embed

        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.decoder = nn.Linear(cfg.hidden, cfg.feature_num)

    def forward(self, input_seqs, masked_pos=None):
        """
        Args:
            input_seqs: [B, T, feature_num]
            masked_pos: [B, N_mask]

        Returns:
            representation: [B, T, hidden]
            logits_lm: [B, N_mask, feature_num] or [B, T, feature_num]
        """
        h = self.transformer(input_seqs)          # [B, T, H]
        representation = h

        if self.output_embed:
            return representation

        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))  # [B, N_mask, H]
            h = torch.gather(h, dim=1, index=masked_pos)                     # [B, N_mask, H]

        h = self.linear(h)
        h = torch.nn.functional.gelu(h)
        h = self.norm(h)
        logits_lm = self.decoder(h)               # [B, N_mask, feature_num]

        return representation, logits_lm


class Contrastive(nn.Module):
    """
    CrossHAR contrastive projection head.

    Input:
        representation: [B, T, hidden]

    Output:
        z: [B, proj_out_dim]
    """
    def __init__(self, cfg):
        super().__init__()

        self.input_dim = cfg.hidden
        # self.seq_transformer_dim = getattr(cfg, "contrastive_hidden", 100)
        # self.seq_transformer_depth = getattr(cfg, "contrastive_depth", 1)
        # self.seq_transformer_heads = getattr(cfg, "contrastive_heads", 4)
        # self.seq_transformer_mlp_dim = getattr(cfg, "contrastive_mlp_dim", 64)
        self.seq_transformer_dim = 100
        self.seq_transformer_depth = 1
        self.seq_transformer_heads = 4
        self.seq_transformer_mlp_dim = 64
        self.dropout = getattr(cfg, "contrastive_dropout", 0.1)

        proj_hidden_dim = getattr(cfg, "projection_hidden_dim", self.input_dim // 2)
        proj_out_dim = getattr(cfg, "projection_out_dim", max(1, self.input_dim // 4))

        self.seq_transformer = Seq_Transformer(
            patch_size=self.input_dim,
            dim=self.seq_transformer_dim,
            depth=self.seq_transformer_depth,
            heads=self.seq_transformer_heads,
            mlp_dim=self.seq_transformer_mlp_dim,
            dropout=self.dropout,
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.seq_transformer_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim)
        )

    def forward(self, representation):
        """
        Args:
            representation: [B, T, H]

        Returns:
            z: [B, D]
        """
        c_t = self.seq_transformer(representation)   # [B, seq_transformer_dim]
        z = self.projection_head(c_t)                # [B, proj_out_dim]
        return z
