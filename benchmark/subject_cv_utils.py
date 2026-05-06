import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from config import TrainConfig, load_mask_config, load_model_config
from utils.utils import get_imu_input_tag, set_seeds, slice_imu_channels, update_model_input_config

ENCODER_BACKBONES = ["transformer", "cnn", "resnet"]

DEFAULT_DATASET_8_NAMES = [
    "DSADS",
    "HARSense",
    "HHAR",
    "KU-HAR",
    "MHEALTH",
    "Motion",
    "PAMAP2",
    "RealWorld",
    "Shoaib",
    "TNDA-HAR",
    "UCI",
    "USC-HAD",
    "UT-Complex",
    "WISDM",
]

PRETRAIN_MODES = [
    "LIMU-BERT",
    "TS-TCC",
    "TS2Vec",
    "SimMTM",
    "BioBankSSL",
    "FOCAL",
    "CrossHAR",
    "CRT",
]


def add_common_subject_cv_args(parser):
    parser.add_argument("--datasets_root", type=str, default="./data")
    parser.add_argument("-ds", "--pretrain_datasets", nargs="+", default=None)
    parser.add_argument("--data_config", type=str, default="./data/data_config.json")
    parser.add_argument("--mode", type=str, default="BioBankSSL", choices=PRETRAIN_MODES)
    parser.add_argument("--encoder_backbone", type=str, default="transformer", choices=ENCODER_BACKBONES)
    parser.add_argument("--model_version", type=str, default="v1")
    parser.add_argument("--fold_id", type=int, default=0, help="0-based fold id")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="Override train config seed for subject split")
    parser.add_argument("--save_dir", type=str, default="./save_subject_cv")
    parser.add_argument("--embed_dir", type=str, default="./embed_subject_cv")
    parser.add_argument("--input_channels", type=int, default=6, choices=[3, 6])
    parser.add_argument("-g", "--gpu", type=str, default=None)
    return parser


def get_pretrain_train_cfg_path(mode):
    return os.path.join("./config", f"{mode}.json")


def build_pretrain_args(args, train_cfg_path=None):
    target = "pretrain_" + args.mode
    model_cfg = load_model_config(
        target,
        args.mode,
        args.model_version,
        encoder_backbone=getattr(args, "encoder_backbone", "transformer"),
    )
    if model_cfg is None:
        raise ValueError(f"Unable to load model config for target={target}, prefix={args.mode}")

    model_cfg = update_model_input_config(model_cfg, args.input_channels)
    fold_save_dir = os.path.join(args.save_dir, f"fold_{args.fold_id}")
    fold_embed_dir = os.path.join(args.embed_dir, f"fold_{args.fold_id}")
    os.makedirs(fold_save_dir, exist_ok=True)
    os.makedirs(fold_embed_dir, exist_ok=True)

    return SimpleNamespace(
        model_version=args.model_version,
        pretrain_datasets=args.pretrain_datasets,
        datasets_root=args.datasets_root,
        data_config=args.data_config,
        dataset_version="raw",
        gpu=args.gpu,
        train_cfg=train_cfg_path or get_pretrain_train_cfg_path(args.mode),
        label_index=-1,
        save_dir=fold_save_dir,
        embed_dir=fold_embed_dir,
        input_channels=args.input_channels,
        model_cfg=model_cfg,
        mode=args.mode,
        encoder_backbone=getattr(args, "encoder_backbone", "transformer"),
        train_subset_rate=1.0,
        data_other_subset_rate=0.0,
        datasets_other_root="./data_other",
        save_path=os.path.join(fold_save_dir, f"{target}_{get_imu_input_tag(args.input_channels)}"),
    )


def load_dataset8_config(data_config_path):
    if not data_config_path or not os.path.exists(data_config_path):
        return {}
    with open(data_config_path, "r") as f:
        return json.load(f)


def dataset8_key(dataset_name):
    return f"{dataset_name.upper()}_20_120"


def label_column_from_config(config, dataset_name, key, fallback):
    ds_cfg = config.get(dataset8_key(dataset_name), {})
    return int(ds_cfg.get(key, fallback))


def get_dataset_names(datasets_root, dataset_names=None):
    if dataset_names:
        return dataset_names

    names = []
    for name in DEFAULT_DATASET_8_NAMES:
        if os.path.exists(os.path.join(datasets_root, name, "data.npy")):
            names.append(name)
    if names:
        return names

    for name in sorted(os.listdir(datasets_root)):
        if os.path.exists(os.path.join(datasets_root, name, "data.npy")):
            names.append(name)
    return names


def load_dataset8_arrays(args, keep_all_label_dims=True):
    config = load_dataset8_config(args.data_config)
    dataset_names = get_dataset_names(args.datasets_root, args.pretrain_datasets)
    if not dataset_names:
        raise ValueError(f"No dataset arrays found under {args.datasets_root}")

    data_list = []
    label_list = []
    loaded_names = []

    for name in dataset_names:
        data_path = os.path.join(args.datasets_root, name, "data.npy")
        label_path = os.path.join(args.datasets_root, name, "label.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        data = np.load(data_path).astype(np.float32)
        labels = np.load(label_path).astype(np.int64)
        data = slice_imu_channels(data, args.input_channels)

        if labels.ndim == 2:
            labels = labels[:, None, :]
        if labels.ndim != 3:
            raise ValueError(f"Unexpected label shape for {name}: {labels.shape}")
        if not keep_all_label_dims:
            activity_idx = label_column_from_config(config, name, "activity_label_index", 0)
            labels = labels[:, :, activity_idx:activity_idx + 1]

        print(f"Loaded [{name}] data={data.shape}, labels={labels.shape}")
        data_list.append(data)
        label_list.append(labels)
        loaded_names.append(name)

    args.pretrain_datasets = loaded_names
    return loaded_names, data_list, label_list


def flatten_label_column(labels, index):
    labels = np.asarray(labels)
    if labels.ndim == 3:
        return labels[:, 0, index].astype(np.int64)
    if labels.ndim == 2:
        return labels[:, index].astype(np.int64)
    raise ValueError(f"Unexpected labels shape: {labels.shape}")


def all_subject_ids(label_list, subject_label_index=1):
    ids = [flatten_label_column(labels, subject_label_index) for labels in label_list]
    return np.unique(np.concatenate(ids, axis=0)).astype(np.int64)


def subject_folds(subject_ids, n_folds=5, seed=None):
    subject_ids = np.sort(np.asarray(subject_ids, dtype=np.int64))
    return [np.sort(x.astype(np.int64)) for x in np.array_split(subject_ids, n_folds)]


def dataset_wise_subject_folds(label_list, n_folds=5, subject_label_index=1):
    """Build folds by splitting subjects inside each dataset, then merging fold ids.

    Fold k contains the k-th subject split from every dataset.  No random
    shuffling is applied; subjects are sorted before np.array_split so the split
    is deterministic and independent of run order.
    """
    merged_folds = [[] for _ in range(n_folds)]
    for labels in label_list:
        subjects = np.unique(flatten_label_column(labels, subject_label_index)).astype(np.int64)
        subjects = np.sort(subjects)
        for fold_id, fold_subjects in enumerate(np.array_split(subjects, n_folds)):
            if fold_subjects.size > 0:
                merged_folds[fold_id].append(fold_subjects.astype(np.int64))

    out = []
    for fold_parts in merged_folds:
        if fold_parts:
            out.append(np.sort(np.unique(np.concatenate(fold_parts, axis=0))).astype(np.int64))
        else:
            out.append(np.array([], dtype=np.int64))
    return out


def get_fold_subject_split(label_list, fold_id, n_folds=5, seed=3431, subject_label_index=1):
    subjects = all_subject_ids(label_list, subject_label_index=subject_label_index)
    folds = dataset_wise_subject_folds(
        label_list,
        n_folds=n_folds,
        subject_label_index=subject_label_index,
    )
    if fold_id < 0 or fold_id >= len(folds):
        raise ValueError(f"fold_id must be in [0, {len(folds) - 1}], got {fold_id}")

    test_subjects = folds[fold_id]
    train_subjects = np.setdiff1d(subjects, test_subjects)
    return train_subjects, test_subjects, folds


def get_dataset_wise_subject_cv_masks(
    labels,
    fold_id,
    n_folds=5,
    subject_label_index=1,
    dataset_label_index=2,
    vali_rate=0.1,
):
    labels = np.asarray(labels)
    subjects = flatten_label_column(labels, subject_label_index).astype(np.int64)
    if labels.ndim >= 2 and dataset_label_index < labels.shape[-1]:
        dataset_ids = flatten_label_column(labels, dataset_label_index).astype(np.int64)
    else:
        dataset_ids = np.zeros(labels.shape[0], dtype=np.int64)

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    vali_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)
    train_subjects_all = []
    vali_subjects_all = []
    test_subjects_all = []

    for dataset_id in np.sort(np.unique(dataset_ids)):
        dataset_mask = dataset_ids == dataset_id
        dataset_subjects = np.sort(np.unique(subjects[dataset_mask]).astype(np.int64))
        folds = subject_folds(dataset_subjects, n_folds=n_folds)
        if fold_id < 0 or fold_id >= len(folds):
            raise ValueError(f"fold_id must be in [0, {len(folds) - 1}], got {fold_id}")

        test_subjects = folds[fold_id]
        train_val_subjects = np.setdiff1d(dataset_subjects, test_subjects)
        vali_count = max(1, int(round(train_val_subjects.size * vali_rate))) if train_val_subjects.size > 1 else 0
        vali_subjects = train_val_subjects[:vali_count]
        train_subjects = train_val_subjects[vali_count:]
        if train_subjects.size == 0:
            train_subjects = vali_subjects
            vali_subjects = np.array([], dtype=np.int64)

        train_mask |= dataset_mask & np.isin(subjects, train_subjects)
        vali_mask |= dataset_mask & np.isin(subjects, vali_subjects)
        test_mask |= dataset_mask & np.isin(subjects, test_subjects)

        train_subjects_all.append(train_subjects)
        vali_subjects_all.append(vali_subjects)
        test_subjects_all.append(test_subjects)

    def merge(parts):
        parts = [x for x in parts if x.size > 0]
        if not parts:
            return np.array([], dtype=np.int64)
        return np.sort(np.unique(np.concatenate(parts, axis=0))).astype(np.int64)

    return (
        train_mask,
        vali_mask,
        test_mask,
        merge(train_subjects_all),
        merge(vali_subjects_all),
        merge(test_subjects_all),
    )


def filter_by_subjects(data_list, label_list, subjects, subject_label_index=1):
    subject_set = set(int(x) for x in subjects)
    out_data = []
    out_labels = []
    for data, labels in zip(data_list, label_list):
        sample_subjects = flatten_label_column(labels, subject_label_index)
        mask = np.array([int(x) in subject_set for x in sample_subjects], dtype=bool)
        out_data.append(data[mask])
        out_labels.append(labels[mask])
    return out_data, out_labels


def split_train_val_subjects(train_subjects, val_rate=0.1, seed=3431):
    train_subjects = np.asarray(train_subjects, dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = train_subjects.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_rate)))
    val_subjects = np.sort(shuffled[:val_count])
    fit_subjects = np.sort(shuffled[val_count:])
    return fit_subjects, val_subjects


def print_fold_summary(fold_id, train_subjects, test_subjects, data_list, label_list, subject_label_index=1):
    train_set = set(int(x) for x in train_subjects)
    test_set = set(int(x) for x in test_subjects)
    if train_set & test_set:
        raise ValueError("Subject leakage detected between train and test folds.")

    train_samples = 0
    test_samples = 0
    for labels in label_list:
        subjects = flatten_label_column(labels, subject_label_index)
        train_samples += int(np.isin(subjects, list(train_set)).sum())
        test_samples += int(np.isin(subjects, list(test_set)).sum())

    print(f"Fold {fold_id}: train_subjects={len(train_subjects)}, test_subjects={len(test_subjects)}")
    print(f"Fold {fold_id}: train_samples={train_samples}, test_samples={test_samples}")


def load_train_and_mask_cfg(args):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if getattr(args, "seed", None) is not None:
        train_cfg = train_cfg._replace(seed=int(args.seed))
    mask_cfg = load_mask_config(args.mode)
    set_seeds(train_cfg.seed)
    return train_cfg, mask_cfg
