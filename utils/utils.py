import argparse
import os
import glob
import torch.nn as nn
import torch
from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, load_mask_config, load_model_config
import random

import numpy as np
import sys

PRETRAIN_MODES = [
    "BioBankSSL",
    "LIMU-BERT",
    "TS-TCC",
    "TS2Vec",
    "SimMTM",
    "FOCAL",
    "CrossHAR",
    "CRT",
]

CLASSIFIER_BACKBONES = [
    "lstm",
    "gru",
    "transformer",
    "mlp",
    "cnn",
    "cnn2",
    "cnn1",
    "cnn2d",
    "cnn1d",
    "dcnn",
    "deepsense",
    "tpn",
    "attn",
]

ENCODER_BACKBONES = ["transformer", "cnn", "resnet"]


def pretrain_config_file_for_mode(mode):
    return f"{mode}.json"


def pretrain_target_for_mode(mode):
    return f"pretrain_{mode}"


def classifier_target_for_mode(mode, classifier_prefix):
    return f"classifier_{mode}_{classifier_prefix}"


def get_imu_input_tag(input_channels):
    input_channels = int(input_channels)
    if input_channels == 3:
        return "acc"
    if input_channels == 6:
        return "imu6"
    raise ValueError(f"Only 3-axis acc or 6-axis IMU are supported, got input_channels={input_channels}")


def slice_imu_channels(data, input_channels):
    input_channels = int(input_channels)
    if input_channels not in (3, 6):
        raise ValueError(f"Only 3-axis acc or 6-axis IMU are supported, got input_channels={input_channels}")
    if data.shape[-1] < input_channels:
        raise ValueError(
            f"Input feature dim {data.shape[-1]} is smaller than requested input_channels={input_channels}"
        )
    return data[:, :, :input_channels]


def update_model_input_config(model_cfg, input_channels):
    input_channels = int(input_channels)
    updates = {}

    if hasattr(model_cfg, "feature_num"):
        updates["feature_num"] = input_channels
    if hasattr(model_cfg, "input"):
        updates["input"] = input_channels

    if hasattr(model_cfg, "focal_use_dual_modalities") and input_channels < 6:
        updates["focal_use_dual_modalities"] = False

    if not updates:
        return model_cfg

    if hasattr(model_cfg, "_replace"):
        return model_cfg._replace(**updates)

    cfg = type("ConfigNamespace", (), {})()
    for key, value in vars(model_cfg).items():
        setattr(cfg, key, value)
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def update_encoder_backbone_config(model_cfg, encoder_backbone):
    if encoder_backbone is None or not hasattr(model_cfg, "encoder_type"):
        return model_cfg

    encoder_backbone = str(encoder_backbone).lower()
    if hasattr(model_cfg, "_replace"):
        return model_cfg._replace(encoder_type=encoder_backbone)

    cfg = type("ConfigNamespace", (), {})()
    for key, value in vars(model_cfg).items():
        setattr(cfg, key, value)
    cfg.encoder_type = encoder_backbone
    return cfg


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


def validate_fraction(name, value, min_exclusive=0.0, max_inclusive=1.0, allow_zero=False):
    value = float(value)
    if allow_zero and value == 0.0:
        return value
    if not (min_exclusive < value <= max_inclusive):
        left = "[" if allow_zero else "("
        raise ValueError(f"{name} must be in {left}{min_exclusive}, {max_inclusive}], got {value}")
    return value


def subsample_training_subset(data, labels, subset_rate=1.0, seed=None, balance=False):
    subset_rate = validate_fraction("train_subset_rate", subset_rate)
    if subset_rate >= 1.0:
        return data, labels

    rng = np.random.default_rng(seed)
    total = data.shape[0]
    target_total = max(1, int(total * subset_rate))

    if balance:
        labels_1d = labels
        if labels_1d.ndim > 1:
            labels_1d = labels_1d.reshape(labels_1d.shape[0], -1)[:, 0]

        indices = []
        unique_labels = np.unique(labels_1d)
        per_class = max(1, target_total // max(len(unique_labels), 1))

        for c in unique_labels:
            cls_idx = np.where(labels_1d == c)[0]
            cls_idx = rng.permutation(cls_idx)
            indices.extend(cls_idx[:min(per_class, len(cls_idx))].tolist())

        if len(indices) < target_total:
            remaining = np.setdiff1d(np.arange(total), np.asarray(indices, dtype=np.int64), assume_unique=False)
            if remaining.size > 0:
                extra = rng.choice(remaining, size=min(target_total - len(indices), remaining.size), replace=False)
                indices.extend(extra.tolist())

        indices = np.asarray(indices[:target_total], dtype=np.int64)
        indices = rng.permutation(indices)
    else:
        indices = rng.choice(total, size=target_total, replace=False)

    return data[indices], labels[indices]


def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                  , training_rate=training_rate, vali_rate=0.2
                                                                                                  , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.2
                                , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.2, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test

def load_classifier_basic_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg

def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index

def _discover_dataset_names(datasets_root):
    if not os.path.exists(datasets_root):
        return []

    names = []
    for name in sorted(os.listdir(datasets_root)):
        dataset_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_dir):
            continue
        if (
            os.path.exists(os.path.join(dataset_dir, "data.npy"))
            and os.path.exists(os.path.join(dataset_dir, "label.npy"))
        ):
            names.append(name)
    return names


def _resolve_pretrain_dataset_paths(datasets_root, dataset_name, dataset_version='20_120'):
    dataset_dir = os.path.join(datasets_root, dataset_name)
    candidates = []
    if dataset_version not in (None, "", "raw"):
        candidates.append((
            os.path.join(dataset_dir, f"data_{dataset_version}.npy"),
            os.path.join(dataset_dir, f"label_{dataset_version}.npy"),
        ))
    candidates.append((
        os.path.join(dataset_dir, "data.npy"),
        os.path.join(dataset_dir, "label.npy"),
    ))

    for data_path, label_path in candidates:
        if os.path.exists(data_path) and os.path.exists(label_path):
            return data_path, label_path
    return candidates[0]


def load_multiple_pretrain_datasets(datasets_root, dataset_names=None, dataset_version='20_120', required=True):
    all_data_list = []
    all_label_list = []

    ref_data_shape = None
    ref_seq_len = None

    if dataset_names is None:
        dataset_names = _discover_dataset_names(datasets_root)
    if not dataset_names:
        if required:
            raise ValueError(f"No datasets found under {datasets_root}")
        return all_data_list, all_label_list

    for name in dataset_names:
        data_path, label_path = _resolve_pretrain_dataset_paths(datasets_root, name, dataset_version)
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            if required:
                raise FileNotFoundError(f'Dataset files not found for [{name}] under {datasets_root}')
            print(f"Skip optional dataset [{name}] under {datasets_root}: files not found.")
            continue

        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path).astype(np.float32)

        print(f'Loaded dataset [{name}] -> data: {data.shape}, labels: {labels.shape}')

        # check data dimension
        if ref_data_shape is None:
            ref_data_shape = data.shape[1:]  # [T,C]
            ref_seq_len = labels.shape[1]
        else:
            if data.shape[1:] != ref_data_shape:
                raise ValueError(
                    f'Data shape mismatch for dataset [{name}]: got {data.shape[1:]}, expected {ref_data_shape}'
                )

            # only check sequence length
            if labels.shape[1] != ref_seq_len:
                raise ValueError(
                    f'Label sequence length mismatch for dataset [{name}]: '
                    f'got {labels.shape[1]}, expected {ref_seq_len}'
                )

        all_data_list.append(data)
        all_label_list.append(labels)

    return all_data_list, all_label_list

# def prepare_multi_pretrain_dataset(data_list, label_list, training_rate=0.8, vali_rate=0.2, seed=None):
#     """
#     Split each dataset independently, then concatenate all train parts and all validation parts.

#     Returns:
#         data_train, label_train, data_vali, label_vali
#     """
#     if seed is not None:
#         set_seeds(seed)

#     train_data_list = []
#     train_label_list = []
#     vali_data_list = []
#     vali_label_list = []

#     for i, (data, labels) in enumerate(zip(data_list, label_list)):
#         if seed is not None:
#             set_seeds(seed + i)

#         data_train, label_train, data_vali, label_vali, _, _ = partition_and_reshape(
#             data,
#             labels,
#             label_index=0,
#             training_rate=training_rate,
#             vali_rate=vali_rate,
#             change_shape=False
#         )

#         print(f'[Dataset {i}] train: {data_train.shape}, vali: {data_vali.shape}')

#         train_data_list.append(data_train)
#         train_label_list.append(label_train)
#         vali_data_list.append(data_vali)
#         vali_label_list.append(label_vali)

#     data_train = np.concatenate(train_data_list, axis=0)
#     label_train = np.concatenate(train_label_list, axis=0)
#     data_vali = np.concatenate(vali_data_list, axis=0)
#     label_vali = np.concatenate(vali_label_list, axis=0)

#     # shuffle after concatenation
#     train_idx = np.arange(data_train.shape[0])
#     np.random.shuffle(train_idx)
#     data_train = data_train[train_idx]
#     label_train = label_train[train_idx]

#     vali_idx = np.arange(data_vali.shape[0])
#     np.random.shuffle(vali_idx)
#     data_vali = data_vali[vali_idx]
#     label_vali = label_vali[vali_idx]

#     print('Multi-dataset pretrain split done:')
#     print(f'  Total train: {data_train.shape}, labels: {label_train.shape}')
#     print(f'  Total vali : {data_vali.shape}, labels: {label_vali.shape}')

#     return data_train, label_train, data_vali, label_vali

def prepare_multi_pretrain_dataset(
    data_list,
    label_list,
    training_rate=0.8,
    vali_rate=0.2,
    seed=None,
    keep_activity_subject=False,  # 新增，默认 False，兼容旧逻辑
    shuffle_samples=True,
):
    """
    Split each dataset independently, then concatenate all train/vali parts.

    If keep_activity_subject=False:
        return label_train/label_vali as original activity-only labels
        (same behavior as before).

    If keep_activity_subject=True:
        return label_train/label_vali as [N, 2] -> [activity, subject].
    """
    if seed is not None:
        set_seeds(seed)

    train_data_list, vali_data_list = [], []
    train_label_list, vali_label_list = [], []

    for i, (data, labels) in enumerate(zip(data_list, label_list)):
        if seed is not None:
            set_seeds(seed + i)

        # Optional shuffle before split.
        arr = np.arange(data.shape[0])
        if shuffle_samples:
            np.random.shuffle(arr)
        data = data[arr]
        labels = labels[arr]

        train_num = int(data.shape[0] * training_rate)
        vali_num = int(data.shape[0] * vali_rate)

        data_train = data[:train_num, ...]
        data_vali = data[train_num:train_num + vali_num, ...]

        labels_train_raw = labels[:train_num, ...]
        labels_vali_raw = labels[train_num:train_num + vali_num, ...]

        if not keep_activity_subject:
            # 完全保持原行为：只返回 activity(label_index=0)
            t = np.min(labels[:, :, 0]) if labels.ndim >= 3 else np.min(labels[:, 0])
            if labels.ndim >= 3:
                label_train = labels_train_raw[:, :, 0] - t
                label_vali = labels_vali_raw[:, :, 0] - t
            else:
                label_train = labels_train_raw[:, 0] - t
                label_vali = labels_vali_raw[:, 0] - t
        else:
            # 返回 [activity, subject]
            if labels.ndim == 3:
                # 取每段第一个时间点标签，通常整段一致
                act_train = labels_train_raw[:, 0, 0]
                sub_train = labels_train_raw[:, 0, 1]
                act_vali = labels_vali_raw[:, 0, 0]
                sub_vali = labels_vali_raw[:, 0, 1]
            elif labels.ndim == 2:
                # [N, D]
                if labels.shape[1] < 2:
                    raise ValueError(
                        f"keep_activity_subject=True requires labels with >=2 columns, got {labels.shape}"
                    )
                act_train = labels_train_raw[:, 0]
                sub_train = labels_train_raw[:, 1]
                act_vali = labels_vali_raw[:, 0]
                sub_vali = labels_vali_raw[:, 1]
            else:
                raise ValueError(f"Unsupported labels shape: {labels.shape}")

            # activity 做平移到从0开始；subject 保留原ID
            act_min = np.min(np.concatenate([act_train, act_vali], axis=0))
            act_train = act_train - act_min
            act_vali = act_vali - act_min

            label_train = np.stack([act_train, sub_train], axis=1).astype(np.int64)
            label_vali = np.stack([act_vali, sub_vali], axis=1).astype(np.int64)

        print(f'[Dataset {i}] train: {data_train.shape}, vali: {data_vali.shape}')

        train_data_list.append(data_train)
        vali_data_list.append(data_vali)
        train_label_list.append(label_train)
        vali_label_list.append(label_vali)

    data_train = np.concatenate(train_data_list, axis=0)
    data_vali = np.concatenate(vali_data_list, axis=0)
    label_train = np.concatenate(train_label_list, axis=0)
    label_vali = np.concatenate(vali_label_list, axis=0)

    # Optional global shuffle
    if shuffle_samples:
        train_idx = np.arange(data_train.shape[0])
        np.random.shuffle(train_idx)
        data_train = data_train[train_idx]
        label_train = label_train[train_idx]

        vali_idx = np.arange(data_vali.shape[0])
        np.random.shuffle(vali_idx)
        data_vali = data_vali[vali_idx]
        label_vali = label_vali[vali_idx]

    print('Multi-dataset pretrain split done:')
    print(f'  Total train: {data_train.shape}, labels: {label_train.shape}')
    print(f'  Total vali : {data_vali.shape}, labels: {label_vali.shape}')

    return data_train, label_train, data_vali, label_vali


def _parse_mode_defaults(default_mode, default_classifier_prefix):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--mode", type=str, default=default_mode)
    bootstrap.add_argument("--classifier_prefix", "--method", dest="classifier_prefix",
                           type=str, default=default_classifier_prefix)
    known, _ = bootstrap.parse_known_args()
    return known.mode, known.classifier_prefix


def handle_argv(
    target=None,
    config_train=None,
    prefix=None,
    task=None,
    default_mode="LIMU-BERT",
    default_classifier_prefix="transformer",
):
    selected_mode, selected_classifier_prefix = _parse_mode_defaults(default_mode, default_classifier_prefix)

    if task is None:
        if target is None:
            task = "classifier" if selected_classifier_prefix else "pretrain"
        elif target.startswith("classifier_"):
            task = "classifier"
        else:
            task = "pretrain"

    if config_train is None:
        config_train = "classifier.json" if task == "classifier" else pretrain_config_file_for_mode(selected_mode)

    parser = argparse.ArgumentParser(description='PyTorch Model')

    parser.add_argument('--mode', type=str, default=selected_mode, choices=PRETRAIN_MODES,
                        help='Self-supervised method / encoder family')

    parser.add_argument('--classifier_prefix', '--method', dest='classifier_prefix',
                        type=str, default=selected_classifier_prefix,
                        choices=CLASSIFIER_BACKBONES,
                        help='Classifier backbone used by classifier scripts')

    parser.add_argument('--model_cfg_path', type=str, default=None,
                        help='Legacy model config JSON fallback')

    parser.add_argument('--encoder_cfg_path', type=str, default='config/encoder.json',
                        help='JSON file containing shared encoder backbone hyperparameters')

    parser.add_argument('--method_cfg_dir', type=str, default='config',
                        help='Directory containing per-method hyperparameter JSON files')

    parser.add_argument('--encoder_backbone', type=str, default='transformer',
                        choices=ENCODER_BACKBONES,
                        help='Encoder backbone selected from CLI; overrides encoder_type in model_cfg_path')

    parser.add_argument('--classifier_cfg_path', type=str, default='config/classifier.json',
                        help='JSON file containing classifier backbone hyperparameters')

    parser.add_argument('--model_version', type=str, default='v1',
                        help='Model config version')

    parser.add_argument('-ds', '--pretrain_datasets', nargs='+', default=None,
                        help='Dataset folder names for joint pretraining, e.g. DSADS Shoaib HHAR')

    parser.add_argument('--datasets_root', type=str, default='./data',
                        help='Root directory containing all datasets')
    parser.add_argument('--datasets_other_root', type=str, default='./data_other',
                        help='Optional root directory containing extra datasets for encoder pretraining')

    parser.add_argument('--dataset_version', type=str, default='20_120',
                        choices=['10_100', '20_120'],
                        help='Dataset version suffix')

    parser.add_argument('-g', '--gpu', type=str, default=None,
                        help='Set specific GPU')

    parser.add_argument('-t', '--train_cfg', type=str,
                        default='./config/' + config_train,
                        help='Training config json file path')

    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    
    parser.add_argument('--save_dir', type=str, default='./saved',
                        help='Directory to save checkpoints')
    parser.add_argument('--embed_dir', type=str, default='./embed_1',
                        help='Directory to save embeddings')
    parser.add_argument('--input_channels', type=int, default=6, choices=[3, 6],
                        help='Use 3-axis accelerometer only or full 6-axis IMU input')
    parser.add_argument('--train_subset_rate', type=float, default=1.0,
                        help='Fraction of the 80%% training split to use, in (0, 1]')
    parser.add_argument('--data_other_subset_rate', type=float, default=0.0,
                        help='Fraction of data_other to add to encoder pretraining, in [0, 1]')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    args.train_subset_rate = validate_fraction("train_subset_rate", args.train_subset_rate)
    args.data_other_subset_rate = validate_fraction(
        "data_other_subset_rate",
        args.data_other_subset_rate,
        allow_zero=True,
    )

    if task == "classifier":
        prefix = args.classifier_prefix if prefix is None else prefix
        target = classifier_target_for_mode(args.mode, args.classifier_prefix) if target is None else target
    else:
        prefix = args.mode if prefix is None else prefix
        target = pretrain_target_for_mode(args.mode) if target is None else target

    model_cfg = load_model_config(
        target,
        prefix,
        args.model_version,
        path_model=args.model_cfg_path,
        path_classifier=args.classifier_cfg_path,
        path_encoder=args.encoder_cfg_path,
        path_method_dir=args.method_cfg_dir,
        encoder_backbone=args.encoder_backbone if task != "classifier" else None,
    )
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()

    args.target = target
    args.mode = args.mode if task != "classifier" else args.mode
    if task != "classifier":
        model_cfg = update_encoder_backbone_config(model_cfg, args.encoder_backbone)
    args.model_cfg = update_model_input_config(model_cfg, args.input_channels)
    os.makedirs(args.save_dir, exist_ok=True)
    args.save_path = os.path.join(args.save_dir, f"{target}_{get_imu_input_tag(args.input_channels)}")

    return args


def handle_pretrain_argv(default_mode="LIMU-BERT"):
    return handle_argv(
        target=None,
        config_train=None,
        prefix=None,
        task="pretrain",
        default_mode=default_mode,
        default_classifier_prefix="transformer",
    )


def handle_embedding_argv(default_mode="BioBankSSL"):
    return handle_pretrain_argv(default_mode=default_mode)


def handle_classifier_argv(default_mode="BioBankSSL", default_classifier_prefix="transformer"):
    return handle_argv(
        target=None,
        config_train="classifier.json",
        prefix=None,
        task="classifier",
        default_mode=default_mode,
        default_classifier_prefix=default_classifier_prefix,
    )

def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels

def load_multi_pretrain_data_config(args):
    model_cfg = update_model_input_config(args.model_cfg, args.input_channels)
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = load_mask_config(args.mode, path_method_dir=args.method_cfg_dir)

    set_seeds(train_cfg.seed)

    data_list, label_list = load_multiple_pretrain_datasets(
        datasets_root=args.datasets_root,
        dataset_names=args.pretrain_datasets,
        dataset_version=args.dataset_version
    )

    data_list = [slice_imu_channels(data, model_cfg.feature_num) for data in data_list]
    args.model_cfg = model_cfg

    return data_list, label_list, train_cfg, model_cfg, mask_cfg

def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
