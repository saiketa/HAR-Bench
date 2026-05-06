import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import trainers.trainer_LIMU_BERT as trainer
from config import TrainConfig, load_model_config
from cross_location_utils import (
    add_common_cross_location_args,
    build_cross_location_pretrain_args,
    canonical_location_name,
    classifier_ckpt_name,
    flatten_label_column,
    get_fold_subject_split,
    load_dataset_arrays,
    mask_by_subjects_and_location,
    pretrain_ckpt_name,
    split_train_val_subjects,
)
from models.classifiers import fetch_classifier
from utils.preprocessors import IMUDataset
from utils.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier for one cross-location CV fold")
    add_common_cross_location_args(parser)
    parser.add_argument("--method", type=str, default="cnn", choices=["gru", "transformer", "mlp", "cnn"])
    parser.add_argument("--classifier_train_cfg", type=str, default="./config/classifier.json")
    parser.add_argument("--training_rate", type=float, default=0.8)
    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--lambda1", type=float, default=6.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    train_location = canonical_location_name(cli_args.train_location)
    test_location = canonical_location_name(cli_args.test_location)
    if train_location == test_location:
        raise ValueError("train_location and test_location must be different for cross-location validation.")

    run_args = build_cross_location_pretrain_args(cli_args)
    ckpt_name = pretrain_ckpt_name(cli_args.mode, cli_args.input_channels, cli_args.lambda1, cli_args.lambda2)
    embedding_path = os.path.join(run_args.embed_dir, f"embedding_{ckpt_name}.npy")
    label_path = os.path.join(run_args.embed_dir, f"label_{ckpt_name}.npy")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    train_cfg = TrainConfig.from_json(cli_args.classifier_train_cfg)
    if cli_args.seed is not None:
        train_cfg = train_cfg._replace(seed=int(cli_args.seed))
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    embeddings = np.load(embedding_path).astype(np.float32)
    labels_full = np.load(label_path).astype(np.float32)
    if embeddings.ndim == 2:
        embeddings = embeddings[:, None, :]
    if labels_full.ndim == 2:
        labels_full = labels_full[:, None, :]

    _, _, label_list = load_dataset_arrays(run_args)
    train_subjects, _, _ = get_fold_subject_split(
        label_list,
        fold_id=cli_args.fold_id,
        n_folds=cli_args.n_folds,
        subject_label_index=cli_args.subject_label_index,
    )
    fit_subjects, val_subjects = split_train_val_subjects(
        train_subjects,
        val_rate=1.0 - cli_args.training_rate,
    )

    activity_labels = flatten_label_column(labels_full, cli_args.activity_label_index).astype(np.int64)
    train_mask = mask_by_subjects_and_location(
        labels_full,
        fit_subjects,
        train_location,
        subject_label_index=cli_args.subject_label_index,
        location_label_index=cli_args.location_label_index,
    )
    val_mask = mask_by_subjects_and_location(
        labels_full,
        val_subjects,
        train_location,
        subject_label_index=cli_args.subject_label_index,
        location_label_index=cli_args.location_label_index,
    )
    if not np.any(val_mask):
        val_mask = train_mask.copy()
    if not np.any(train_mask):
        raise ValueError(f"No classifier training samples for train_location={train_location}, fold={cli_args.fold_id}")

    data_train_pool = embeddings[train_mask]
    label_train_pool = activity_labels[train_mask]
    data_vali = embeddings[val_mask]
    label_vali = activity_labels[val_mask]

    if cli_args.label_rate < 1.0:
        if cli_args.balance:
            selected = []
            unique_labels = np.unique(label_train_pool)
            target_total = max(1, int(len(data_train_pool) * cli_args.label_rate))
            per_class = max(1, target_total // len(unique_labels))
            for c in unique_labels:
                cls_idx = np.where(label_train_pool == c)[0]
                np.random.shuffle(cls_idx)
                selected.extend(cls_idx[:min(per_class, len(cls_idx))].tolist())
            selected = np.asarray(selected, dtype=np.int64)
        else:
            selected = np.random.choice(
                len(data_train_pool),
                max(1, int(len(data_train_pool) * cli_args.label_rate)),
                replace=False,
            )
        data_train = data_train_pool[selected]
        label_train = label_train_pool[selected]
    else:
        data_train = data_train_pool
        label_train = label_train_pool

    model_cfg = load_model_config(
        f"classifier_{cli_args.mode}_{cli_args.method}",
        cli_args.method,
        cli_args.model_version,
    )
    if model_cfg is None:
        raise ValueError(f"Unable to load classifier config: {cli_args.method}")
    if hasattr(model_cfg, "_replace"):
        model_cfg = model_cfg._replace(seq_len=int(data_train.shape[1]))
    else:
        model_cfg = SimpleNamespace(**vars(model_cfg))
        model_cfg.seq_len = int(data_train.shape[1])

    print(f"Classifier train_location={train_location}, test_location={test_location}, fold={cli_args.fold_id}")
    print("Train embedding shape:", data_train.shape)
    print("Val embedding shape:", data_vali.shape)

    label_num = int(np.max(activity_labels)) + 1
    model = fetch_classifier(cli_args.method, model_cfg, input=data_train.shape[-1], output=label_num)
    if model is None:
        raise ValueError(f"Unsupported classifier method: {cli_args.method}")

    train_set = IMUDataset(data_train, label_train, feature_len=data_train.shape[-1], pipeline=[], isInstanceNorm=False)
    vali_set = IMUDataset(data_vali, label_vali, feature_len=data_vali.shape[-1], pipeline=[], isInstanceNorm=False)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=train_cfg.batch_size)
    vali_loader = DataLoader(vali_set, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    save_path = os.path.join(
        run_args.save_dir,
        classifier_ckpt_name(cli_args.mode, cli_args.method, cli_args.input_channels),
    )
    finetune_trainer = trainer.Trainer(train_cfg, model, optimizer, save_path, get_device(cli_args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        return criterion(model(inputs, True), label)

    def func_forward(model, batch):
        inputs, label = batch
        return model(inputs, False), label

    def func_evaluate(label, predicts):
        y_true = label.cpu().numpy()
        y_pred = torch.argmax(predicts, dim=1).cpu().numpy()
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

    finetune_trainer.train(func_loss, func_forward, func_evaluate, train_loader, vali_loader, vali_loader)


if __name__ == "__main__":
    main()
