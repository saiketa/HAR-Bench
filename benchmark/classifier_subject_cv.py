import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import trainers.trainer_LIMU_BERT as trainer
from config import load_model_config
from subject_cv_utils import (
    add_common_subject_cv_args,
    flatten_label_column,
    get_dataset_wise_subject_cv_masks,
)
from utils.preprocessors import IMUDataset
from utils.utils import get_device, get_imu_input_tag
from models.classifiers import fetch_classifier
from config import TrainConfig


def train_subject_cv_classifier(
    args,
    data,
    labels,
    label_index,
    training_rate,
    label_rate,
    balance,
    method,
    fold_id,
    n_folds,
    subject_label_index,
    dataset_label_index,
):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if getattr(args, "seed", None) is not None:
        train_cfg = train_cfg._replace(seed=int(args.seed))

    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    if data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim != 3:
        raise ValueError(f"Unexpected embedding shape: {data.shape}")

    labels_raw = labels
    labels = flatten_label_column(labels_raw, label_index).astype(np.int64)
    train_mask, vali_mask, test_mask, train_subjects, vali_subjects, test_subjects = (
        get_dataset_wise_subject_cv_masks(
            labels_raw,
            fold_id=fold_id,
            n_folds=n_folds,
            subject_label_index=subject_label_index,
            dataset_label_index=dataset_label_index,
            vali_rate=1.0 - training_rate,
        )
    )

    data_train_all = data[train_mask]
    label_train_all = labels[train_mask]
    data_vali = data[vali_mask]
    label_vali = labels[vali_mask]
    if data_vali.shape[0] == 0:
        data_vali = data_train_all
        label_vali = label_train_all

    print(f"Dataset-wise subject CV fold: {fold_id}/{n_folds}")
    print(f"Train subjects: {train_subjects.tolist()}")
    print(f"Val subjects: {vali_subjects.tolist()}")
    print(f"Held-out test subjects: {test_subjects.tolist()}")
    print(f"Held-out test samples: {int(test_mask.sum())}")

    if label_rate < 1.0:
        if balance:
            unique_labels = np.unique(label_train_all)
            train_indices = []
            target_total = max(1, int(len(data_train_all) * label_rate))
            per_class = max(1, target_total // len(unique_labels))
            for c in unique_labels:
                cls_idx = np.where(label_train_all == c)[0]
                np.random.shuffle(cls_idx)
                train_indices.extend(cls_idx[:min(per_class, len(cls_idx))].tolist())
            train_indices = np.array(train_indices, dtype=np.int64)
        else:
            num_labeled = max(1, int(len(data_train_all) * label_rate))
            train_indices = np.random.choice(len(data_train_all), num_labeled, replace=False)
        data_train = data_train_all[train_indices]
        label_train = label_train_all[train_indices]
    else:
        data_train = data_train_all
        label_train = label_train_all

    print("Train embedding shape:", data_train.shape)
    print("Val embedding shape:", data_vali.shape)
    print("Train labels shape:", label_train.shape)
    print("Val labels shape:", label_vali.shape)

    model_cfg = args.model_cfg
    seq_len_for_classifier = data_train.shape[1]
    if hasattr(model_cfg, "_replace"):
        model_cfg = model_cfg._replace(seq_len=int(seq_len_for_classifier))
    else:
        model_cfg = SimpleNamespace(**vars(model_cfg))
        model_cfg.seq_len = int(seq_len_for_classifier)

    data_set_train = IMUDataset(data_train, label_train, feature_len=data_train.shape[-1], pipeline=[], isInstanceNorm=False)
    data_set_vali = IMUDataset(data_vali, label_vali, feature_len=data_vali.shape[-1], pipeline=[], isInstanceNorm=False)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)

    label_num = int(np.max(labels)) + 1
    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
    if model is None:
        raise ValueError(f"Unsupported classifier method: {method}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    finetune_trainer = trainer.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        return criterion(logits, label)

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        y_true = label.cpu().numpy()
        y_pred = torch.argmax(predicts, dim=1).cpu().numpy()
        return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

    finetune_trainer.train(
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_vali,
        data_loader_vali,
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier for one cross-subject CV fold")
    add_common_subject_cv_args(parser)
    parser.add_argument("--method", type=str, default="transformer", choices=["gru", "transformer", "mlp", "cnn"])
    parser.add_argument("--classifier_train_cfg", type=str, default="./config/classifier.json")
    parser.add_argument("--training_rate", type=float, default=0.8)
    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--lambda1", type=float, default=6.0, help="CrossHAR loss weight lambda1")
    parser.add_argument("--lambda2", type=float, default=1.0, help="CrossHAR loss weight lambda2")
    parser.add_argument("--activity_label_index", type=int, default=0)
    parser.add_argument("--subject_label_index", type=int, default=1)
    parser.add_argument("--dataset_label_index", type=int, default=2)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    input_tag = get_imu_input_tag(cli_args.input_channels)
    fold_save_dir = os.path.join(cli_args.save_dir, f"fold_{cli_args.fold_id}")
    fold_embed_dir = os.path.join(cli_args.embed_dir, f"fold_{cli_args.fold_id}")

    model_cfg = load_model_config(
        f"classifier_{cli_args.mode}_{cli_args.method}",
        cli_args.method,
        cli_args.model_version,
    )
    if model_cfg is None:
        raise ValueError(
            f"Unable to load classifier config for mode={cli_args.mode}, method={cli_args.method}"
        )

    if cli_args.mode == "CrossHAR":
        pretrain_name = f"pretrain_{cli_args.mode}_{input_tag}_masked_{cli_args.lambda1}_{cli_args.lambda2}"
    else:
        pretrain_name = f"pretrain_{cli_args.mode}_{input_tag}"
    embedding_path = os.path.join(fold_embed_dir, f"embedding_{pretrain_name}.npy")
    label_path = os.path.join(fold_embed_dir, f"label_{pretrain_name}.npy")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    os.makedirs(fold_save_dir, exist_ok=True)
    run_args = SimpleNamespace(
        train_cfg=cli_args.classifier_train_cfg,
        model_cfg=model_cfg,
        gpu=cli_args.gpu,
        seed=cli_args.seed,
        save_path=os.path.join(
            fold_save_dir,
            f"classifier_{cli_args.mode}_{cli_args.method}_{input_tag}",
        ),
    )

    embedding_data = np.load(embedding_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)
    dataset_id_path = os.path.join(fold_embed_dir, f"dataset_id_{pretrain_name}.npy")
    if labels.ndim == 2:
        labels = labels[:, None, :]
    if labels.ndim == 3 and labels.shape[-1] <= cli_args.dataset_label_index and os.path.exists(dataset_id_path):
        dataset_ids = np.load(dataset_id_path).astype(np.float32)
        dataset_label = np.broadcast_to(
            dataset_ids[:, None, None],
            (labels.shape[0], labels.shape[1], 1),
        )
        labels = np.concatenate([labels, dataset_label], axis=-1)

    train_subject_cv_classifier(
        args=run_args,
        data=embedding_data,
        labels=labels,
        label_index=cli_args.activity_label_index,
        training_rate=cli_args.training_rate,
        label_rate=cli_args.label_rate,
        balance=cli_args.balance,
        method=cli_args.method,
        fold_id=cli_args.fold_id,
        n_folds=cli_args.n_folds,
        subject_label_index=cli_args.subject_label_index,
        dataset_label_index=cli_args.dataset_label_index,
    )


if __name__ == "__main__":
    main()
