# models/CRT.py
import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .LIMU_BERT import Transformer as LIMUTransformer
from .base_models import resnet1d18
from .encoders import ResNet1DEncoder, CNN1DEncoder


class cnn_extractor(nn.Module):
    def __init__(self, dim, input_plane):
        super().__init__()
        self.cnn = resnet1d18(input_channels=dim, inplanes=input_plane)

    def forward(self, x):
        return self.cnn(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, seq_len=None):
        super().__init__()
        if seq_len is None:
            raise ValueError("seq_len is required for LIMU_BERT positional embedding")

        cfg = SimpleNamespace(
            feature_num=dim,
            hidden=dim,
            seq_len=seq_len,
            n_layers=depth,
            n_heads=heads,
            hidden_ff=mlp_dim,
            dropout=dropout,
            emb_norm=True,
            p_drop_attn=dropout,
            p_drop_hidden=dropout,
            p_drop_emb=dropout,
            eps=1e-12,
        )
        self.net = LIMUTransformer(cfg)

    def forward(self, x):
        return self.net(x)


def build_crt_encoder_token_mixer(cfg, dim, depth, heads, mlp_dim, seq_len, dropout=0.0, dim_head=64):
    encoder_type = str(getattr(cfg, "encoder_type", "transformer")).lower()
    if encoder_type == "transformer":
        return Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            seq_len=seq_len,
        )

    mixer_cfg = SimpleNamespace(
        feature_num=dim,
        hidden=dim,
        seq_len=seq_len,
        encoder_type=encoder_type,
        resnet_blocks=int(getattr(cfg, "resnet_blocks", depth)),
        resnet_kernel_size=int(getattr(cfg, "resnet_kernel_size", 5)),
        resnet_dropout=float(getattr(cfg, "resnet_dropout", dropout)),
        resnet_dilations=list(getattr(cfg, "resnet_dilations", [1, 2, 4, 8])),
        cnn_depth=int(getattr(cfg, "cnn_depth", depth)),
        cnn_kernel_size=int(getattr(cfg, "cnn_kernel_size", 5)),
        cnn_dropout=float(getattr(cfg, "cnn_dropout", dropout)),
        cnn_dilations=list(getattr(cfg, "cnn_dilations", [1, 2, 4, 8])),
        cnn_hidden_multiplier=int(getattr(cfg, "cnn_hidden_multiplier", 2)),
    )

    if encoder_type == "resnet":
        return ResNet1DEncoder(mixer_cfg)
    if encoder_type == "cnn":
        return CNN1DEncoder(mixer_cfg)
    raise ValueError(f"Unsupported CRT encoder_type: {encoder_type}")


class TFR(nn.Module):
    def __init__(self, seq_len, patch_len, num_classes, dim, depth, heads, mlp_dim, cfg, channels=12,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        assert seq_len % (4 * patch_len) == 0, "seq_len should be 4 * n * patch_len"

        self.patch_len = patch_len
        self.patch_token_count = seq_len // patch_len
        self.total_token_count = self.patch_token_count + 3

        self.to_patch = nn.Sequential(
            Rearrange("b c (n p1) -> b n c p1", p1=patch_len),
            Rearrange("b n c p1 -> (b n) c p1")
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_token_count, dim))
        self.modal_embedding = nn.Parameter(torch.randn(3, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 3, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = build_crt_encoder_token_mixer(
            cfg=cfg,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            seq_len=self.total_token_count,
        )

        self.cnn1 = cnn_extractor(dim=channels, input_plane=dim // 8)
        self.cnn2 = cnn_extractor(dim=channels, input_plane=dim // 8)
        self.cnn3 = cnn_extractor(dim=channels, input_plane=dim // 8)

    def forward(self, x):
        batch, _, time_steps = x.shape
        t = x[:, :, :time_steps // 2]
        m = x[:, :, time_steps // 2: time_steps * 3 // 4]
        p = x[:, :, -time_steps // 4:]

        t, m, p = self.to_patch(t), self.to_patch(m), self.to_patch(p)

        patch2seq = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("(b n) c 1 -> b n c", b=batch)
        )

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=batch)
        x = torch.cat(
            (
                cls_tokens[:, 0:1, :], patch2seq(self.cnn1(t)),
                cls_tokens[:, 1:2, :], patch2seq(self.cnn2(m)),
                cls_tokens[:, 2:3, :], patch2seq(self.cnn3(p)),
            ),
            dim=1,
        )

        token_len = x.shape[1]
        ts = token_len - 3
        t_token_idx, m_token_idx, p_token_idx = 0, ts // 2 + 1, ts * 3 // 4 + 2

        x[:, :m_token_idx] += self.modal_embedding[:1]
        x[:, m_token_idx:p_token_idx] += self.modal_embedding[1:2]
        x[:, p_token_idx:] += self.modal_embedding[2:]
        x += self.pos_embedding[:, :token_len]
        x = self.dropout(x)
        x = self.transformer(x)

        t_token, m_token, p_token = x[:, t_token_idx], x[:, m_token_idx], x[:, p_token_idx]
        return (t_token + m_token + p_token) / 3


def TFR_Encoder(cfg):
    return TFR(
        seq_len=cfg.seq_len,
        patch_len=cfg.patch_len,
        num_classes=getattr(cfg, "num_classes", 2),
        dim=cfg.hidden,
        depth=6,
        heads=8,
        mlp_dim=cfg.hidden,
        dropout=0.2,
        emb_dropout=0.1,
        channels=cfg.feature_num,
        cfg=cfg,
    )


class CRT(nn.Module):
    def __init__(self, encoder, decoder_dim, decoder_depth=2, decoder_heads=8,
                 decoder_dim_head=64, patch_len=20, in_dim=12):
        super().__init__()
        self.encoder = encoder
        self.patch_len = patch_len

        self.total_token_count = encoder.pos_embedding.shape[1]
        self.patch_token_count = self.total_token_count - 3

        self.to_patch = encoder.to_patch
        pixel_values_per_patch = in_dim * patch_len

        self.modal_embedding = self.encoder.modal_embedding
        self.mask_token = nn.Parameter(torch.randn(3, decoder_dim))

        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim,
            seq_len=self.total_token_count,
        )

        self.decoder_pos_emb = nn.Embedding(self.patch_token_count, decoder_dim)
        self.to_pixels = nn.ModuleList([nn.Linear(decoder_dim, pixel_values_per_patch) for _ in range(3)])
        self.projs = nn.ModuleList([nn.Linear(decoder_dim, decoder_dim) for _ in range(2)])

    def IDC_loss(self, tokens, encoded_tokens):
        _, t, _ = tokens.shape
        tokens = F.normalize(tokens, dim=-1)
        encoded_tokens = F.normalize(encoded_tokens, dim=-1).transpose(2, 1)
        cross_mul = torch.exp(torch.matmul(tokens, encoded_tokens))
        mask = (1 - torch.eye(t, device=tokens.device)).unsqueeze(0)
        cross_mul = cross_mul * mask
        return torch.log(cross_mul.sum(-1).sum(-1).clamp_min(1e-8)).mean(-1)

    def forward(self, x, mask_ratio=0.75, beta=1e-4):
        device = x.device
        patches = self.to_patch[0](x)
        batch, num_patches, _, _ = patches.shape

        num_masked = int(mask_ratio * num_patches)

        rand_indices1 = torch.randperm(num_patches // 2, device=device)
        masked_indices1 = rand_indices1[: num_masked // 2].sort()[0]
        unmasked_indices1 = rand_indices1[num_masked // 2:].sort()[0]

        rand_indices2 = torch.randperm(num_patches // 4, device=device)
        masked_indices2 = rand_indices2[: num_masked // 4].sort()[0]
        unmasked_indices2 = rand_indices2[num_masked // 4:].sort()[0]

        rand_indices = torch.cat((
            masked_indices1, unmasked_indices1,
            masked_indices2 + num_patches // 2, unmasked_indices2 + num_patches // 2,
            masked_indices2 + num_patches // 4 * 3, unmasked_indices2 + num_patches // 4 * 3
        ))

        masked_num_t = masked_indices1.shape[0]
        masked_num_f = 2 * masked_indices2.shape[0]

        tpatches = patches[:, : num_patches // 2, :, :]
        mpatches = patches[:, num_patches // 2: num_patches * 3 // 4, :, :]
        ppatches = patches[:, -num_patches // 4:, :, :]

        unmasked_tpatches = tpatches[:, unmasked_indices1, :, :]
        unmasked_mpatches = mpatches[:, unmasked_indices2, :, :]
        unmasked_ppatches = ppatches[:, unmasked_indices2, :, :]

        t_tokens = self.to_patch[1](unmasked_tpatches)
        m_tokens = self.to_patch[1](unmasked_mpatches)
        p_tokens = self.to_patch[1](unmasked_ppatches)

        t_tokens = self.encoder.cnn1(t_tokens)
        m_tokens = self.encoder.cnn2(m_tokens)
        p_tokens = self.encoder.cnn3(p_tokens)

        flat = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("(b n) c 1 -> b n c", b=batch)
        )
        t_tokens, m_tokens, p_tokens = flat(t_tokens), flat(m_tokens), flat(p_tokens)
        ori_tokens = torch.cat((t_tokens, m_tokens, p_tokens), 1).clone()

        cls_tokens = repeat(self.encoder.cls_token, "() n d -> b n d", b=batch)
        tokens = torch.cat(
            (
                cls_tokens[:, 0:1, :], t_tokens,
                cls_tokens[:, 1:2, :], m_tokens,
                cls_tokens[:, 2:3, :], p_tokens,
            ),
            dim=1,
        )

        t_idx, m_idx = num_patches // 2 - 1, num_patches * 3 // 4 - 1
        pos_embedding = torch.cat((
            self.encoder.pos_embedding[:, 0:1, :],
            self.encoder.pos_embedding[:, unmasked_indices1 + 1, :],
            self.encoder.pos_embedding[:, t_idx + 2: t_idx + 3],
            self.encoder.pos_embedding[:, unmasked_indices2 + t_idx + 3, :],
            self.encoder.pos_embedding[:, m_idx + 3: m_idx + 4],
            self.encoder.pos_embedding[:, unmasked_indices2 + m_idx + 4, :],
        ), dim=1)

        # 用真实 token 数量，避免长度错位
        um_t = t_tokens.shape[1]
        um_m = m_tokens.shape[1]
        um_p = p_tokens.shape[1]
        modal_embedding = torch.cat((
            repeat(self.modal_embedding[0], "1 d -> 1 n d", n=um_t + 1),
            repeat(self.modal_embedding[1], "1 d -> 1 n d", n=um_m + 1),
            repeat(self.modal_embedding[2], "1 d -> 1 n d", n=um_p + 1),
        ), dim=1)

        if tokens.shape[1] != pos_embedding.shape[1] or tokens.shape[1] != modal_embedding.shape[1]:
            raise RuntimeError(
                f"Token mismatch: tokens={tokens.shape}, pos={pos_embedding.shape}, modal={modal_embedding.shape}"
            )

        tokens = tokens + pos_embedding + modal_embedding
        encoded_tokens = self.encoder.transformer(tokens)

        t_idx = um_t
        m_idx = um_m + um_t + 1

        encoded_wo_cls = torch.cat((
            encoded_tokens[:, 1:t_idx + 1],
            encoded_tokens[:, t_idx + 2:m_idx + 1],
            encoded_tokens[:, m_idx + 2:],
        ), dim=1)

        idc_loss = self.IDC_loss(self.projs[0](ori_tokens), self.projs[1](encoded_wo_cls))

        decoder_tokens = encoded_tokens

        mask_tokens1 = repeat(self.mask_token[0], "d -> b n d", b=batch, n=masked_num_t)
        mask_tokens2 = repeat(self.mask_token[1], "d -> b n d", b=batch, n=masked_num_f // 2)
        mask_tokens3 = repeat(self.mask_token[2], "d -> b n d", b=batch, n=masked_num_f // 2)
        mask_tokens = torch.cat((mask_tokens1, mask_tokens2, mask_tokens3), dim=1)

        decoder_pos_emb = self.decoder_pos_emb(torch.cat((
            masked_indices1,
            masked_indices2 + num_patches // 2,
            masked_indices2 + num_patches * 3 // 4
        )))

        mask_tokens = mask_tokens + decoder_pos_emb
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # Use decoded tokens and explicit layout split to avoid off-by-one slice drift.
        encoded_len = encoded_tokens.shape[1]
        decoded_unmasked = decoded_tokens[:, :encoded_len]
        decoded_masked = decoded_tokens[:, encoded_len:]

        # [t_cls][t_tokens][m_cls][m_tokens][p_cls][p_tokens]
        dec_t = decoded_unmasked[:, 1:1 + um_t]
        dec_m = decoded_unmasked[:, 2 + um_t:2 + um_t + um_m]
        dec_p = decoded_unmasked[:, 3 + um_t + um_m:3 + um_t + um_m + um_p]

        mt = masked_num_t
        mm = masked_num_f // 2
        mp = masked_num_f // 2
        dec_mt = decoded_masked[:, :mt]
        dec_mm = decoded_masked[:, mt:mt + mm]
        dec_mp = decoded_masked[:, mt + mm:mt + mm + mp]

        pred_pixel_values_t = self.to_pixels[0](torch.cat((dec_t, dec_mt), dim=1))
        pred_pixel_values_m = self.to_pixels[1](torch.cat((dec_m, dec_mm), dim=1))
        pred_pixel_values_p = self.to_pixels[2](torch.cat((dec_p, dec_mp), dim=1))

        pred_pixel_values = torch.cat((pred_pixel_values_t, pred_pixel_values_m, pred_pixel_values_p), dim=1)
        target_pixel_values = rearrange(patches[:, rand_indices], "b n c p -> b n (c p)")

        if pred_pixel_values.shape[1] != target_pixel_values.shape[1]:
            raise RuntimeError(
                f"Recon token mismatch: pred={pred_pixel_values.shape}, target={target_pixel_values.shape}, "
                f"(um_t,um_m,um_p)=({um_t},{um_m},{um_p}), (mt,mm,mp)=({mt},{mm},{mp}), num_patches={num_patches}"
            )

        recon_loss = F.mse_loss(pred_pixel_values, target_pixel_values)

        total = recon_loss + beta * idc_loss
        return {"loss": total, "recon_loss": recon_loss, "idc_loss": idc_loss}


class CRT4Pretrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.expected_seq_len = cfg.seq_len
        self.core = CRT(
            encoder=TFR_Encoder(cfg),
            decoder_dim=cfg.hidden,
            patch_len=cfg.patch_len,
            in_dim=cfg.feature_num,
        )

    def forward(self, x, mask_ratio=0.5, beta=1e-4):
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, L, C], got {tuple(x.shape)}")
        if x.shape[1] != self.expected_seq_len:
            raise ValueError(f"Input seq len mismatch: got {x.shape[1]}, expected {self.expected_seq_len}")
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        return self.core(x, mask_ratio=mask_ratio, beta=beta)

    def backbone_features(self, x):
        """
        Feature extraction API for embedding.py.
        Input:  x [B, L, C] where L is seq_len after Preprocess4CRT (typically 2T)
        Output: [B, D]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, L, C], got {tuple(x.shape)}")
        if x.shape[1] != self.expected_seq_len:
            raise ValueError(f"Input seq len mismatch: got {x.shape[1]}, expected {self.expected_seq_len}")
        x = x.transpose(1, 2)  # [B, L, C] -> [B, C, L]
        return self.core.encoder(x)
