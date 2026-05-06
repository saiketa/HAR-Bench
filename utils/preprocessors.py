import argparse
import os
import glob
import torch.nn as nn
import torch
from scipy.special import factorial
from scipy import interpolate
from torch.utils.data import Dataset
from scipy.stats import special_ortho_group
try:
    from tsai.data.core import TSTensor
    from tsai.data.transforms import TSMagWarp, TSTimeWarp
except ImportError:
    TSTensor = None
    TSMagWarp = None
    TSTimeWarp = None

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config
from .augmentations import DataTransform
import random
from .utils import span_mask

import numpy as np
import sys


def _cubic_spline_interpolation(x_eval, x_data, y_data):
    cubic_spline = interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def time_warp_transform(x, sigma=0.2, num_knots=4):
    time_stamps = np.arange(x.shape[1])
    knot_xs = (
        np.arange(0, num_knots + 2, dtype=float) * (x.shape[1] - 1) / (num_knots + 1)
    )
    spline_ys = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0] * x.shape[2], num_knots + 2)
    )

    spline_values = np.array(
        [
            _cubic_spline_interpolation(time_stamps, knot_xs, spline_ys_individual)
            for spline_ys_individual in spline_ys
        ]
    )

    cumulative_sum = np.cumsum(spline_values, axis=1)
    distorted_time_stamps_all = (
        cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (x.shape[1] - 1)
    )

    x_transformed = np.empty(shape=x.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        x_transformed[i // x.shape[2], :, i % x.shape[2]] = np.interp(
            time_stamps, distorted_time_stamps, x[i // x.shape[2], :, i % x.shape[2]]
        )
    return x_transformed

class IMUDataset(Dataset):
    """Load labeled IMU sequences with instance normalization"""
    def __init__(self, data, labels, feature_len=6, pipeline=[], isInstanceNorm=True):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.feature_len = feature_len
        self.isInstanceNorm = isInstanceNorm
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        if self.isInstanceNorm:
            self.data = torch.tensor(self.data.transpose((0, 2, 1)), dtype=torch.float32)
            self.data = self.instance_norm(self.data)
            self.data = self.data.numpy().transpose((0, 2, 1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return (
            torch.from_numpy(instance).float(),
            torch.tensor(self.labels[index]).long()
        )

    def __len__(self):
        return len(self.data)

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError

class Preprocess4Rotation(Pipeline):

    def __init__(self, sensor_dimen=3):
        super().__init__()
        self.sensor_dimen = sensor_dimen

    def __call__(self, instance):
        return self.rotate_random(instance)

    def rotate_random(self, instance):
        instance_new = instance.reshape(instance.shape[0], instance.shape[1] // self.sensor_dimen, self.sensor_dimen)
        rotation_matrix = special_ortho_group.rvs(self.sensor_dimen)
        for i in range(instance_new.shape[1]):
            instance_new[:, i, :] = np.dot(instance_new[:, i, :], rotation_matrix)
        return instance_new.reshape(instance.shape[0], instance.shape[1])
    
class Preprocess4Sample(Pipeline):

    def __init__(self, seq_len, temporal=0.4, temporal_range=[0.8, 1.2]):
        super().__init__()
        self.seq_len = seq_len
        self.temporal = temporal
        self.temporal_range = temporal_range

    def __call__(self, instance):
        if instance.shape[0] == self.seq_len:
            return instance
        if self.temporal > 0:
            temporal_prob = np.random.random()
            if temporal_prob < self.temporal:
                x = np.arange(instance.shape[0])
                ratio_random = np.random.random() * (self.temporal_range[1] - self.temporal_range[0]) + self.temporal_range[0]
                seq_len_scale = int(np.round(ratio_random * self.seq_len))
                index_rand = np.random.randint(0, high=instance.shape[0] - seq_len_scale)
                instance_new = np.zeros((self.seq_len, instance.shape[1]))
                for i in range(instance.shape[1]):
                    f = interpolate.interp1d(x, instance[:, i], kind='linear')
                    x_new = index_rand + np.linspace(0, seq_len_scale, self.seq_len)
                    instance_new[:, i] = f(x_new)
                return instance_new
        index_rand = np.random.randint(0, high=instance.shape[0] - self.seq_len)
        return instance[index_rand:index_rand + self.seq_len, :]

class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg, full_sequence=False):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob
        self.full_sequence = full_sequence

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        if self.full_sequence:
            full_pos = np.arange(shape[0], dtype=np.int64)
            seq = instance.copy()
            return instance_mask, full_pos, np.array(seq)

        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)

class Preprocess4MultiTask:
    """
    Output:
        x, aot_y, permute_y, time_w_y
    """
    def __init__(
        self,
        feature_len=6,
        permute_segments=4,
        time_warp_sigma=0.2,
        p_reverse=0.5,
        p_permute=0.5,
        p_time_warp=0.5,
        permutation_min_seg_length=10,
        return_tensor=False,
    ):
        self.feature_len = feature_len
        self.permute_segments = permute_segments
        self.time_warp_sigma = time_warp_sigma
        self.p_reverse = p_reverse
        self.p_permute = p_permute
        self.p_time_warp = p_time_warp
        self.permutation_min_seg_length = permutation_min_seg_length
        self.return_tensor = return_tensor

    def _time_reversal(self, x):
        return x[::-1].copy()

    def _permutation(self, x):
        T = x.shape[0]
        n_perm = int(self.permute_segments)
        if n_perm <= 1 or n_perm > T:
            return x.copy()

        min_seg_length = int(self.permutation_min_seg_length)
        if T <= min_seg_length * n_perm:
            min_seg_length = max(1, (T - 1) // max(n_perm, 1))
        if min_seg_length <= 0:
            return x.copy()

        idx = np.random.permutation(n_perm)
        while True:
            segs = np.zeros(n_perm + 1, dtype=int)
            low = min_seg_length
            high = T - min_seg_length
            if high <= low:
                return x.copy()
            segs[1:-1] = np.sort(np.random.randint(low, high, n_perm - 1))
            segs[-1] = T
            if np.min(segs[1:] - segs[:-1]) > min_seg_length:
                break

        x_new = np.zeros_like(x)
        pp = 0
        for ii in range(n_perm):
            x_temp = x[segs[idx[ii]]:segs[idx[ii] + 1], :]
            x_new[pp:pp + len(x_temp), :] = x_temp
            pp += len(x_temp)
        return x_new.astype(np.float32)

    def _generate_random_curves(self, x, sigma=0.2, knot=4):
        if x.shape[1] != 3:
            raise ValueError(f"BioBankSSL time warp expects tri-axial blocks, got {x.shape[1]} channels.")
        xx = (
            np.ones((x.shape[1], 1))
            * (np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))
        ).transpose()
        yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, x.shape[1]))
        x_range = np.arange(x.shape[0])
        cs_x = interpolate.CubicSpline(xx[:, 0], yy[:, 0])
        cs_y = interpolate.CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = interpolate.CubicSpline(xx[:, 2], yy[:, 2])
        return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

    def _time_warp_triaxial(self, x):
        tt = self._generate_random_curves(x, sigma=self.time_warp_sigma)
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [
            (x.shape[0] - 1) / tt_cum[-1, 0],
            (x.shape[0] - 1) / tt_cum[-1, 1],
            (x.shape[0] - 1) / tt_cum[-1, 2],
        ]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
        tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]

        x_new = np.zeros(x.shape)
        x_range = np.arange(x.shape[0])
        x_new[:, 0] = np.interp(x_range, tt_cum[:, 0], x[:, 0])
        x_new[:, 1] = np.interp(x_range, tt_cum[:, 1], x[:, 1])
        x_new[:, 2] = np.interp(x_range, tt_cum[:, 2], x[:, 2])
        return x_new.astype(np.float32)

    def _time_warp(self, x):
        if x.shape[1] == 3:
            return self._time_warp_triaxial(x)
        if x.shape[1] == 6:
            acc = self._time_warp_triaxial(x[:, :3])
            gyro = self._time_warp_triaxial(x[:, 3:6])
            return np.concatenate([acc, gyro], axis=1).astype(np.float32)
        return x.copy().astype(np.float32)

    def __call__(self, instance):
        x = instance.astype(np.float32).copy()

        if np.random.rand() < self.p_reverse:
            x = self._time_reversal(x)
            aot_y = 1
        else:
            aot_y = 0

        if np.random.rand() < self.p_permute:
            x = self._permutation(x)
            permute_y = 1
        else:
            permute_y = 0

        if np.random.rand() < self.p_time_warp:
            x = self._time_warp(x)
            time_w_y = 1
        else:
            time_w_y = 0

        if self.return_tensor:
            return (
                torch.from_numpy(x).float(),
                torch.tensor(aot_y).long(),
                torch.tensor(permute_y).long(),
                torch.tensor(time_w_y).long(),
            )

        return (
            x.astype(np.float32),
            np.int64(aot_y),
            np.int64(permute_y),
            np.int64(time_w_y),
        )

class Preprocess4Aug:
    def __init__(self, feature_len=6, return_tensor=False):
        self.feature_len = feature_len
        self.return_tensor = return_tensor

    def __call__(self, instance):
        batch_data = np.expand_dims(instance, axis=0)   
        aug1, aug2 = DataTransform(batch_data)      

        aug1 = aug1[0]
        aug2 = aug2[0]

        if self.return_tensor:
            aug1 = torch.from_numpy(aug1).float()
            aug2 = torch.from_numpy(aug2).float()

        return aug1, aug2


class Preprocess4FOCAL:
    """
    FOCAL-style random augmentation for sequence-sampled IMU data.

    Input:
        instance: [S, T, C], where C=6 and channels are
            0:3 -> acceleration, 3:6 -> gyroscope.
    Output:
        aug1, aug2: two independently augmented FFT views, both [S, T, 2*C].
    """

    def __init__(
        self,
        feature_len=6,
        scaling_sigma=0.2,
        time_warp_sigma=0.2,
        time_warp_knots=6,
        mag_warp_sigma=0.05,
        mag_warp_knots=4,
        time_aug_prob=0.5,
        phase_shift_prob=0.5,
    ):
        self.feature_len = feature_len
        self.modality_names = ["acc", "gyro"] if int(feature_len) >= 6 else ["acc"]
        self.scaling_sigma = scaling_sigma
        self.time_warp_sigma = time_warp_sigma
        self.time_warp_knots = time_warp_knots
        self.mag_warp_sigma = mag_warp_sigma
        self.mag_warp_knots = mag_warp_knots
        self.time_aug_prob = time_aug_prob
        self.phase_shift_prob = phase_shift_prob
        self.time_warp_func = (
            TSTimeWarp(magnitude=self.time_warp_sigma, order=self.time_warp_knots)
            if TSTimeWarp is not None else None
        )
        self.mag_warp_func = (
            TSMagWarp(magnitude=self.mag_warp_sigma, order=self.mag_warp_knots)
            if TSMagWarp is not None else None
        )

        self.time_augmenters = [
            "permutation",
            "negation",
            "time_warp",
            "horizontal_flip",
            "mag_warp",
            "scaling",
        ]
        self.freq_augmenters = ["phase_shift"]
        self.aug_names = self.time_augmenters + self.freq_augmenters

    def _split_modalities(self, x):
        if x.shape[-1] < 3:
            raise ValueError(f"FOCAL preprocess expects at least 3 channels, got {x.shape[-1]}")
        if x.shape[-1] >= 6 and len(self.modality_names) == 2:
            return {
                "acc": x[..., 0:3],
                "gyro": x[..., 3:6],
            }
        return {
            "acc": x[..., 0:3],
        }

    def _merge_modalities(self, mods):
        return np.concatenate([mods[name] for name in self.modality_names], axis=-1).astype(np.float32)

    def _fft_modalities(self, mods):
        freq_mods = {}
        for name, x_mod in mods.items():
            z = np.fft.fft(x_mod, axis=1)
            freq_mods[name] = np.stack([z.real, z.imag], axis=-1).reshape(
                x_mod.shape[0],
                x_mod.shape[1],
                x_mod.shape[2] * 2,
            ).astype(np.float32)
        return freq_mods

    def _phase_shift_freq_domain(self, x_mod):
        # x_mod: [S, T, 2*Cmod], packed as real/imag pairs per original channel.
        s, t, c2 = x_mod.shape
        x_complex = x_mod.reshape(s, t, c2 // 2, 2)
        z = x_complex[..., 0] + 1j * x_complex[..., 1]
        angle = np.random.uniform(-np.pi, np.pi)
        z = np.abs(z) * np.exp(1j * (np.angle(z) + angle))
        return np.stack([z.real, z.imag], axis=-1).reshape(s, t, c2).astype(np.float32)

    def _phase_shift_time_domain(self, x_mod):
        # Kept for backward compatibility with old checkpoints/scripts.
        z = np.fft.fft(x_mod, axis=1)
        angle = np.random.uniform(-np.pi, np.pi)
        z = np.abs(z) * np.exp(1j * (np.angle(z) + angle))
        out = np.fft.ifft(z, axis=1).real
        return out.astype(np.float32)

    def _tsai_warp(self, x_mod, warp_func):
        # Official FOCAL applies tsai warps on [batch, channel, time].
        if warp_func is None or TSTensor is None:
            return None

        x_tensor = torch.from_numpy(x_mod.transpose(0, 2, 1)).float()
        warped = warp_func(TSTensor(x_tensor), split_idx=0).data
        return warped.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)

    def _mag_warp(self, x_mod):
        # x_mod: [S, T, Cmod]
        warped = self._tsai_warp(x_mod, self.mag_warp_func)
        if warped is not None:
            return warped

        s, t, c = x_mod.shape
        x_steps = np.arange(t, dtype=np.float32)
        knot_x = np.linspace(0, t - 1, self.mag_warp_knots + 2, dtype=np.float32)
        out = np.empty_like(x_mod, dtype=np.float32)

        for i in range(s):
            for j in range(c):
                knot_y = np.random.normal(loc=1.0, scale=self.mag_warp_sigma, size=(self.mag_warp_knots + 2,))
                curve = np.interp(x_steps, knot_x, knot_y).astype(np.float32)
                out[i, :, j] = x_mod[i, :, j] * curve
        return out

    def _apply_time_augment(self, mods, aug_name):
        out = {name: mods[name].copy() for name in self.modality_names}

        if aug_name == "permutation":
            perm = np.random.permutation(out["acc"].shape[0])
            for k in out:
                if random.random() < self.time_aug_prob:
                    out[k] = out[k][perm]

        elif aug_name == "negation":
            for k in out:
                if random.random() < self.time_aug_prob:
                    out[k] = -out[k]

        elif aug_name == "horizontal_flip":
            for k in out:
                if random.random() < self.time_aug_prob:
                    out[k] = np.flip(out[k], axis=(0, 1)).copy()

        elif aug_name == "scaling":
            for k in out:
                if random.random() < self.time_aug_prob:
                    scale = np.random.normal(loc=1.0, scale=self.scaling_sigma)
                    out[k] = out[k] * np.float32(scale)

        elif aug_name == "time_warp":
            for k in out:
                if random.random() < self.time_aug_prob:
                    warped = self._tsai_warp(out[k], self.time_warp_func)
                    if warped is None:
                        warped = time_warp_transform(
                            out[k],
                            sigma=self.time_warp_sigma,
                            num_knots=self.time_warp_knots,
                        ).astype(np.float32)
                    out[k] = warped

        elif aug_name == "mag_warp":
            for k in out:
                if random.random() < self.time_aug_prob:
                    out[k] = self._mag_warp(out[k]).astype(np.float32)

        return out

    def _apply_freq_augment(self, mods, aug_name):
        out = {name: mods[name].copy() for name in self.modality_names}
        if aug_name == "phase_shift":
            for k in out:
                if random.random() < self.phase_shift_prob:
                    out[k] = self._phase_shift_freq_domain(out[k])
        return out

    def _random_augment_once(self, instance):
        aug_name = random.choice(self.aug_names)
        return self._augment_with_name(instance, aug_name)

    def _augment_with_name(self, instance, aug_name):
        mods = self._split_modalities(instance.astype(np.float32))

        if aug_name in self.time_augmenters:
            mods = self._apply_time_augment(mods, aug_name)
            mods = self._fft_modalities(mods)
        elif aug_name in self.freq_augmenters:
            mods = self._fft_modalities(mods)
            mods = self._apply_freq_augment(mods, aug_name)
        else:
            raise ValueError(f"Unsupported FOCAL augmentation: {aug_name}")

        return self._merge_modalities(mods)

    def augment_batch(self, batch_instance, aug_name=None):
        """
        batch_instance: [B, S, T, C]
        Apply one augmenter for the whole batch (same as original FOCAL random mode).
        """
        if aug_name is None:
            aug_name = random.choice(self.aug_names)
        outs = [self._augment_with_name(batch_instance[i], aug_name) for i in range(batch_instance.shape[0])]
        return np.stack(outs, axis=0).astype(np.float32), aug_name

    def __call__(self, instance):
        x = instance.astype(np.float32).copy()
        aug1 = self._random_augment_once(x)
        aug2 = self._random_augment_once(x)
        return aug1, aug2

class Preprocess4CrossHAR:
    """
    CrossHAR pretraining preprocess:
    raw instance -> two augmented views -> masked reconstruction targets

    Output:
        mask_seq_1, masked_pos_1, seq_1,
        mask_seq_2, masked_pos_2, seq_2
    """
    def __init__(self, mask_cfg, return_tensor=False):
        self.mask_ratio = mask_cfg.mask_ratio
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob
        self.return_tensor = return_tensor

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def _mask_one_view(self, instance):
        """
        Apply span masking to one augmented view.
        instance: [T, C]
        """
        shape = instance.shape
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))
        mask_pos = span_mask(shape[0], self.max_gram, goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            rand = np.random.rand()
            if rand < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif rand < self.mask_prob + self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            rand = np.random.rand()
            if rand < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif rand < self.mask_prob + self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))

        seq = instance[mask_pos_index, :]

        return (
            instance_mask.astype(np.float32),
            np.array(mask_pos_index),
            np.array(seq, dtype=np.float32)
        )

    def __call__(self, instance):
        """
        instance: [T, C]
        """
        batch_data = np.expand_dims(instance, axis=0)   # [1, T, C]
        aug1, aug2 = DataTransform(batch_data)          # each: [1, T, C]
        aug1, aug2 = aug1[0], aug2[0]

        mask_seq_1, masked_pos_1, seq_1 = self._mask_one_view(aug1)
        mask_seq_2, masked_pos_2, seq_2 = self._mask_one_view(aug2)

        if self.return_tensor:
            return (
                torch.from_numpy(mask_seq_1).float(),
                torch.from_numpy(masked_pos_1).long(),
                torch.from_numpy(seq_1).float(),
                torch.from_numpy(mask_seq_2).float(),
                torch.from_numpy(masked_pos_2).long(),
                torch.from_numpy(seq_2).float(),
            )

        return (
            mask_seq_1,
            masked_pos_1,
            seq_1,
            mask_seq_2,
            masked_pos_2,
            seq_2,
        )

class Preprocess4CRT:
    def __init__(self, feature_len=6, return_tensor=False):
        self.feature_len = feature_len
        self.return_tensor = return_tensor

    def __call__(self, instance):
        # instance: [T, C]
        time = instance.astype(np.float32).copy()
        freq = np.fft.fft(time, axis=0)[: time.shape[0] // 2, :]

        a = freq.real
        b = freq.imag

        magnitude = np.abs(freq).astype(np.float32)

        phase = np.zeros_like(a, dtype=np.float32)
        pos = a > 0
        neg = a < 0
        zer = a == 0
        phase[pos] = np.arctan(b[pos] / a[pos])
        phase[neg] = np.arctan(b[neg] / a[neg]) + np.sign(b[neg]) * np.pi
        phase[zer] = np.sign(b[zer]) * np.pi / 2

        out = np.concatenate([time, magnitude, phase], axis=0).astype(np.float32)  # [2T, C]
        return torch.from_numpy(out).float() if self.return_tensor else out
