import argparse
import os
import glob
import torch.nn as nn
import torch
from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config
from .augmentations import DataTransform
import random

import numpy as np
import sys

class Dataset4Contrast(Dataset):
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        mean = np.mean(self.data, axis=1)
        var = np.var(self.data, axis=1)
        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)

        aug1, aug2 = instance
        return torch.from_numpy(aug1).float(), torch.from_numpy(aug2).float()

    def __len__(self):
        return len(self.data)


class Dataset4TS2Vec(Dataset):
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float() if not torch.is_tensor(instance) else instance.float()

    def __len__(self):
        return len(self.data)


class Dataset4SimMTM(Dataset):
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float() if not torch.is_tensor(instance) else instance.float()

    def __len__(self):
        return len(self.data)


class Dataset4FOCAL(Dataset):
    """
    Sequence-sampler style dataset for FOCAL pretraining.

    Returns:
        subseq: [S, T, C], where S=focal_seq_len.
    """

    def __init__(self, data, feature_len, seq_len=4, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.seq_len = int(seq_len)
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

        self.subseq_indices = []
        n = len(self.data)
        for start in range(0, n, self.seq_len):
            idx = list(range(start, min(start + self.seq_len, n)))
            while len(idx) < self.seq_len:
                idx.append(idx[-1])
            self.subseq_indices.append(idx)

    def __getitem__(self, index):
        idx = self.subseq_indices[index]
        instance = self.data[idx].astype(np.float32)  # [S, T, C]

        for proc in self.pipeline:
            instance = proc(instance)

        return torch.from_numpy(instance).float() if not torch.is_tensor(instance) else instance.float()

    def __len__(self):
        return len(self.subseq_indices)

class Dataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        mean = np.mean(self.data, axis=1)
        var = np.var(self.data, axis=1)
        self.data = torch.tensor(self.data.transpose((0,2,1)))
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0,2,1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return len(self.data)

class Dataset4MultiTask(Dataset):
    """
    Returns:
        x, aot_y, permute_y, time_w_y
    """
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]

        for proc in self.pipeline:
            instance = proc(instance)

        x, aot_y, permute_y, time_w_y = instance

        return (
            torch.from_numpy(x).float(),
            torch.tensor(aot_y).long(),
            torch.tensor(permute_y).long(),
            torch.tensor(time_w_y).long(),
        )

    def __len__(self):
        return len(self.data)

class Dataset4CrossHAR(Dataset):
    """
    Returns:
        mask_seq_1, masked_pos_1, seq_1,
        mask_seq_2, masked_pos_2, seq_2
    """
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.data = torch.tensor(
            self.data.transpose((0, 2, 1)),
            dtype=torch.float32
        )
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]

        for proc in self.pipeline:
            instance = proc(instance)

        mask_seq_1, masked_pos_1, seq_1, \
        mask_seq_2, masked_pos_2, seq_2 = instance

        return (
            torch.from_numpy(mask_seq_1).float(),
            torch.from_numpy(masked_pos_1).long(),
            torch.from_numpy(seq_1).float(),
            torch.from_numpy(mask_seq_2).float(),
            torch.from_numpy(masked_pos_2).long(),
            torch.from_numpy(seq_2).float(),
        )

    def __len__(self):
        return len(self.data)

class Dataset4CRT(Dataset):
    def __init__(self, data, feature_len, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.feature_len = feature_len
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        # instance norm (same style as other benchmark datasets)
        self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)  # [N, C, T]
        self.data = self.instance_norm(self.data)
        self.data = self.data.numpy().transpose((0, 2, 1))  # [N, T, C]

    def __getitem__(self, index):
        instance = self.data[index]  # [T, C]

        for proc in self.pipeline:
            instance = proc(instance)  # expected [2T, C] after Preprocess4CRT

        return instance.float() if torch.is_tensor(instance) else torch.from_numpy(instance).float()

    def __len__(self):
        return len(self.data)
