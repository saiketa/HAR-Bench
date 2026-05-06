import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from config import TrainConfig, load_mask_config, load_model_config
from utils.utils import get_imu_input_tag, set_seeds, slice_imu_channels, update_model_input_config


ENCODER_BACKBONES = ["transformer", "cnn", "resnet"]

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

DEFAULT_DATASET_NAMES = [
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

LOCATION_GROUPS = {
    "trunk": [0, 8, 9, 12],
    "torso": [0, 8, 9, 12],
    "upper": [1, 6, 11],
    "lower": [2, 3, 4, 5, 7],
}

CANONICAL_LOCATION_NAMES = {
    "trunk": "trunk",
    "torso": "trunk",
    "upper": "upper",
    "lower": "lower",
}


def canonical_location_name(name):
    name = str(name).lower()
    if name not in LOCATION_GROUPS:
        raise ValueError(f"Unknown location group: {name}. Choose from trunk, upper, lower.")
    return CANONICAL_LOCATION_NAMES[name]


def location_ids(name):
    return np.asarray(LOCATION_GROUPS[canonical_location_name(name)], dtype=np.int64)


def add_common_cross_location_args(parser):
    parser.add_argument("--datasets_root", type=str, default="./data")
    parser.add_argument("-ds", "--pretrain_datasets", nargs="+", default=None)
    parser.add_argument("--data_config", type=str, default="./data/data_config.json")
    parser.add_argument("--mode", type=str, default="BioBankSSL", choices=PRETRAIN_MODES)
    parser.add_argument("--encoder_backbone", type=str, default="transformer", choices=ENCODER_BACKBONES)
    parser.add_argument("--model_version", type=str, default="v1")
    parser.add_argument("--fold_id", type=int, default=0, help="0-based fold id")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--train_location", type=str, default="trunk", choices=["trunk", "torso", "upper", "lower"])
    parser.add_argument("--test_location", type=str, default="upper", choices=["trunk", "torso", "upper", "lower"])
    parser.add_argument("--activity_label_index", type=int, default=0)
    parser.add_argument("--subject_label_index", type=int, default=1)
    parser.add_argument("--location_label_index", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./save_cross_location")
    parser.add_argument("--embed_dir", type=str, default="./embed_cross_location")
    parser.add_argument("--input_channels", type=int, default=6, choices=[3, 6])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-g", "--gpu", type=str, default=None)
    return parser


def get_pretrain_train_cfg_path(mode):
    return os.path.join("./config", f"{mode}.json")


def location_run_name(train_location, test_location):
    return f"train_{canonical_location_name(train_location)}_test_{canonical_location_name(test_location)}"


def build_cross_location_pretrain_args(args, train_cfg_path=None):
    train_location = canonical_location_name(args.train_location)
    test_location = canonical_location_name(args.test_location)
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

    run_name = location_run_name(train_location, test_location)
    fold_save_dir = os.path.join(args.save_dir, run_name, f"fold_{args.fold_id}")
    fold_embed_dir = os.path.join(args.embed_dir, run_name, f"fold_{args.fold_id}")
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


def get_dataset_names(datasets_root, dataset_names=None):
    if dataset_names:
        return dataset_names

    names = []
    for name in DEFAULT_DATASET_NAMES:
        if os.path.exists(os.path.join(datasets_root, name, "data.npy")):
            names.append(name)
    if names:
        return names

    for name in sorted(os.listdir(datasets_root)):
        if os.path.exists(os.path.join(datasets_root, name, "data.npy")):
            names.append(name)
    return names


def load_dataset_config(data_config_path):
    if not data_config_path or not os.path.exists(data_config_path):
        return {}
    with open(data_config_path, "r") as f:
        return json.load(f)


def load_dataset_arrays(args):
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
        if labels.shape[-1] <= args.location_label_index:
            raise ValueError(
                f"{name} labels must contain activity, subject, and body-location columns; got {labels.shape}"
            )

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
    return np.sort(np.unique(np.concatenate(ids, axis=0))).astype(np.int64)


def subject_folds(subject_ids, n_folds=5):
    subject_ids = np.sort(np.asarray(subject_ids, dtype=np.int64))
    return [np.sort(x.astype(np.int64)) for x in np.array_split(subject_ids, n_folds)]


def get_fold_subject_split(label_list, fold_id, n_folds=5, subject_label_index=1):
    subjects = all_subject_ids(label_list, subject_label_index=subject_label_index)
    folds = subject_folds(subjects, n_folds=n_folds)
    if fold_id < 0 or fold_id >= len(folds):
        raise ValueError(f"fold_id must be in [0, {len(folds) - 1}], got {fold_id}")
    test_subjects = folds[fold_id]
    train_subjects = np.setdiff1d(subjects, test_subjects)
    return train_subjects, test_subjects, folds


def split_train_val_subjects(train_subjects, val_rate=0.2):
    train_subjects = np.sort(np.asarray(train_subjects, dtype=np.int64))
    if train_subjects.size <= 1:
        return train_subjects, np.array([], dtype=np.int64)
    val_count = max(1, int(round(train_subjects.size * val_rate)))
    val_subjects = train_subjects[:val_count]
    fit_subjects = train_subjects[val_count:]
    if fit_subjects.size == 0:
        fit_subjects = val_subjects
        val_subjects = np.array([], dtype=np.int64)
    return fit_subjects, val_subjects


def mask_by_subjects_and_location(labels, subjects, location_name, subject_label_index=1, location_label_index=2):
    sample_subjects = flatten_label_column(labels, subject_label_index)
    sample_locations = flatten_label_column(labels, location_label_index)
    return np.isin(sample_subjects, np.asarray(subjects, dtype=np.int64)) & np.isin(
        sample_locations,
        location_ids(location_name),
    )


def filter_by_subjects_and_location(
    data_list,
    label_list,
    subjects,
    location_name,
    subject_label_index=1,
    location_label_index=2,
):
    out_data = []
    out_labels = []
    for data, labels in zip(data_list, label_list):
        mask = mask_by_subjects_and_location(
            labels,
            subjects,
            location_name,
            subject_label_index=subject_label_index,
            location_label_index=location_label_index,
        )
        out_data.append(data[mask])
        out_labels.append(labels[mask])
    return out_data, out_labels


def filter_usable_datasets(dataset_names, data_list, label_list, training_rate=0.8):
    usable = []
    skipped = []
    for name, data, labels in zip(dataset_names, data_list, label_list):
        train_num = int(data.shape[0] * training_rate)
        vali_num = int(data.shape[0] * (1.0 - training_rate))
        if data.shape[0] == 0 or train_num == 0 or vali_num == 0:
            skipped.append((name, int(data.shape[0]), train_num, vali_num))
            continue
        usable.append((name, data, labels))
    return usable, skipped


def load_train_and_mask_cfg(args):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if getattr(args, "seed", None) is not None:
        train_cfg = train_cfg._replace(seed=int(args.seed))
    mask_cfg = load_mask_config(args.mode)
    set_seeds(train_cfg.seed)
    return train_cfg, mask_cfg


def pretrain_ckpt_name(mode, input_channels, lambda1=6.0, lambda2=1.0):
    input_tag = get_imu_input_tag(input_channels)
    name = f"pretrain_{mode}_{input_tag}"
    if mode == "CrossHAR":
        name = f"{name}_masked_{lambda1}_{lambda2}"
    return name


def classifier_ckpt_name(mode, method, input_channels):
    return f"classifier_{mode}_{method}_{get_imu_input_tag(input_channels)}"


def print_location_fold_summary(
    fold_id,
    train_location,
    test_location,
    train_subjects,
    test_subjects,
    data_list,
    label_list,
    subject_label_index=1,
    location_label_index=2,
):
    train_samples = 0
    test_samples = 0
    for data, labels in zip(data_list, label_list):
        train_samples += int(
            mask_by_subjects_and_location(
                labels,
                train_subjects,
                train_location,
                subject_label_index=subject_label_index,
                location_label_index=location_label_index,
            ).sum()
        )
        test_samples += int(
            mask_by_subjects_and_location(
                labels,
                test_subjects,
                test_location,
                subject_label_index=subject_label_index,
                location_label_index=location_label_index,
            ).sum()
        )

    if set(int(x) for x in train_subjects) & set(int(x) for x in test_subjects):
        raise ValueError("Subject leakage detected between train and test folds.")
    print(
        f"Fold {fold_id}: train_location={train_location}, test_location={test_location}, "
        f"train_subjects={len(train_subjects)}, test_subjects={len(test_subjects)}"
    )
    print(f"Fold {fold_id}: train_samples={train_samples}, test_samples={test_samples}")
