import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Seq_Transformer
from .encoders import build_sequence_encoder


class TC(nn.Module):
    """
    Temporal Contrastive module for TS-TCC.

    Input:
        features_aug1: [B, C, T]
        features_aug2: [B, C, T]

    Output:
        nce: scalar contrastive loss
        c_t: contextual representation, shape [B, tc_hidden]
    """
    def __init__(self, bb_dim, device, tc_hidden=100, timestep=6, temp_unit='tsfm'):
        super(TC, self).__init__()
        self.num_channels = bb_dim
        self.timestep = timestep
        self.device = device
        self.temp_unit = temp_unit

        self.Wk = nn.ModuleList([
            nn.Linear(tc_hidden, self.num_channels) for _ in range(self.timestep)
        ])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.projection_head = nn.Sequential(
            nn.Linear(tc_hidden, self.num_channels // 2),
            nn.BatchNorm1d(self.num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_channels // 2, self.num_channels // 4),
        )

        if self.temp_unit == 'tsfm':
            self.seq_transformer = Seq_Transformer(
                patch_size=self.num_channels,
                dim=tc_hidden,
                depth=1,
                heads=4,
                mlp_dim=64
            )
        elif self.temp_unit == 'lstm':
            self.lstm = nn.LSTM(
                input_size=self.num_channels,
                hidden_size=tc_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
        elif self.temp_unit == 'blstm':
            self.blstm = nn.LSTM(
                input_size=self.num_channels,
                hidden_size=tc_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        elif self.temp_unit == 'gru':
            self.gru = nn.GRU(
                input_size=self.num_channels,
                hidden_size=tc_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
        elif self.temp_unit == 'bgru':
            self.bgru = nn.GRU(
                input_size=self.num_channels,
                hidden_size=tc_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
        else:
            raise ValueError(f"Unsupported temp_unit: {self.temp_unit}")

    def forward(self, features_aug1, features_aug2):
        """
        Args:
            features_aug1: [B, C, T]
            features_aug2: [B, C, T]
        Returns:
            nce: scalar
            c_t: [B, tc_hidden]
        """
        z_aug1 = features_aug1
        z_aug2 = features_aug2

        seq_len = z_aug1.shape[2]
        batch = z_aug1.shape[0]

        # convert to [B, T, C]
        z_aug1 = z_aug1.transpose(1, 2)
        z_aug2 = z_aug2.transpose(1, 2)

        # avoid invalid randint upper bound
        max_start = seq_len - self.timestep
        if max_start <= 0:
            raise ValueError(
                f"seq_len={seq_len} is too short for timestep={self.timestep}. "
                f"Need seq_len > timestep."
            )

        t_samples = torch.randint(
            low=0,
            high=max_start,
            size=(1,),
            device=self.device
        ).long()

        nce = 0.0

        # future samples from aug2
        encode_samples = torch.empty(
            (self.timestep, batch, self.num_channels),
            device=self.device,
            dtype=torch.float32
        )

        for i in np.arange(1, self.timestep + 1):
            idx = (t_samples + i).long()
            encode_samples[i - 1] = z_aug2[:, idx, :].view(batch, self.num_channels)

        # context from aug1
        forward_seq = z_aug1[:, :t_samples + 1, :]   # [B, t+1, C]

        if self.temp_unit == 'tsfm':
            c_t = self.seq_transformer(forward_seq)   # [B, tc_hidden]
        elif self.temp_unit == 'lstm':
            _, (c_t, _) = self.lstm(forward_seq)
            c_t = torch.squeeze(c_t, dim=0)
        elif self.temp_unit == 'blstm':
            _, (c_t, _) = self.blstm(forward_seq)
            c_t = c_t[0, :, :]
        elif self.temp_unit == 'gru':
            _, c_t = self.gru(forward_seq)
            c_t = torch.squeeze(c_t, dim=0)
        elif self.temp_unit == 'bgru':
            _, c_t = self.bgru(forward_seq)
            c_t = c_t[0, :, :]

        pred = torch.empty(
            (self.timestep, batch, self.num_channels),
            device=self.device,
            dtype=torch.float32
        )

        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], pred[i].transpose(0, 1))   # [B, B]
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce /= -1.0 * batch * self.timestep

        return nce, self.projection_head(c_t)


class TSTCC4Pretrain(nn.Module):
    """
    Base model for TS-TCC pretraining.

    This model follows the interface of original TS-TCC base_Model:
        output = model(x)
        output -> (predictions, features)

    where:
        predictions: dummy classification logits, shape [B, num_classes]
        features: backbone features for temporal contrast, shape [B, C, T]
    """
    def __init__(self, cfg, num_classes=None):
        super().__init__()
        self.transformer = build_sequence_encoder(cfg)

        if num_classes is None:
            num_classes = getattr(cfg, "num_classes", 2)

        self.num_classes = num_classes
        self.classifier = nn.Linear(cfg.hidden, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, T, feature_num]

        Returns:
            predictions: [B, num_classes]
            features: [B, hidden, T]
        """
        h = self.transformer(x)          # [B, T, H]

        pooled = h.mean(dim=1)           # [B, H]
        predictions = self.classifier(pooled)

        # TC expects [B, C, T]
        features = h.transpose(1, 2)     # [B, H, T]

        return predictions, features
