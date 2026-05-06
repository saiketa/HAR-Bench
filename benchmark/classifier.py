import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

import trainers.trainer_LIMU_BERT as trainer
from models.classifiers import fetch_classifier
from utils.utils import (
    get_device,
    handle_classifier_argv,
    get_imu_input_tag,
    subsample_training_subset,
)
from utils.preprocessors import IMUDataset
from config import TrainConfig


def classify_embeddings(
    args,
    data,
    labels,
    label_index,
    training_rate,
    label_rate,
    balance=False,
    method=None,
):
    train_cfg = TrainConfig.from_json(args.train_cfg)
    if getattr(args, "seed", None) is not None:
        train_cfg = train_cfg._replace(seed=int(args.seed))
    model_cfg = args.model_cfg

    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    # -------- adapt embedding shape --------
    # allow [N, D] or [N, T, D]
    if data.ndim == 2:
        data = data[:, None, :]
    elif data.ndim == 3:
        pass
    else:
        raise ValueError(
            f"Unexpected embedding shape: {data.shape}, expected [N, D] or [N, T, D]"
        )

    # -------- adapt label shape --------
    # allow [N, 1, K], [N, 1], [N]
    if labels.ndim == 3:
        if label_index >= labels.shape[-1]:
            raise ValueError(
                f"label_index={label_index} out of range for label shape {labels.shape}"
            )
        labels = labels[:, 0, label_index]
    elif labels.ndim == 2:
        labels = labels[:, 0]
    elif labels.ndim == 1:
        pass
    else:
        raise ValueError(f"Unexpected label shape: {labels.shape}")

    labels = labels.astype(np.int64)

    # -------- split train / val --------
    data_train_all, data_vali, label_train_all, label_vali = train_test_split(
        data,
        labels,
        test_size=1.0 - training_rate,
        random_state=train_cfg.seed,
        stratify=labels
    )

    # -------- sample requested subset from the fixed 80% training split --------
    data_train, label_train = subsample_training_subset(
        data_train_all,
        label_train_all,
        subset_rate=label_rate,
        seed=train_cfg.seed,
        balance=balance,
    )
    print(f"Using {label_rate:.3f} of the 80% classifier training split.")

    print("Train embedding shape:", data_train.shape)
    print("Val embedding shape:", data_vali.shape)
    print("Train labels shape:", label_train.shape)
    print("Val labels shape:", label_vali.shape)

    seq_len_for_classifier = data_train.shape[1]
    if hasattr(model_cfg, "_replace"):
        model_cfg = model_cfg._replace(seq_len=int(seq_len_for_classifier))
    else:
        model_cfg = SimpleNamespace(**vars(model_cfg))
        model_cfg.seq_len = int(seq_len_for_classifier)

    pipeline = []
    data_set_train = IMUDataset(
        data_train,
        label_train,
        feature_len=data_train.shape[-1],
        pipeline=pipeline,
        isInstanceNorm=False
    )
    data_set_vali = IMUDataset(
        data_vali,
        label_vali,
        feature_len=data_vali.shape[-1],
        pipeline=pipeline,
        isInstanceNorm=False
    )

    data_loader_train = DataLoader(
        data_set_train,
        shuffle=True,
        batch_size=train_cfg.batch_size
    )
    data_loader_vali = DataLoader(
        data_set_vali,
        shuffle=False,
        batch_size=train_cfg.batch_size
    )

    label_num = len(np.unique(labels))

    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier(
        method,
        model_cfg,
        input=data_train.shape[-1],
        output=label_num
    )

    if model is None:
        raise ValueError(f"Unsupported classifier method: {method}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)

    finetune_trainer = trainer.Trainer(
        train_cfg,
        model,
        optimizer,
        args.save_path,
        get_device(args.gpu)
    )

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        y_true = label.cpu().numpy()
        y_pred = torch.argmax(predicts, dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        return acc, f1

    # reuse existing trainer API
    finetune_trainer.train(
        func_loss,
        func_forward,
        func_evaluate,
        data_loader_train,
        data_loader_vali,
        data_loader_vali
    )

    return model


def resolve_embedding_and_label_paths(mode, lambda1=6.0, lambda2=1.0, embed_dir="embed", input_channels=6):
    """
    Resolve saved embedding / label path by pretrain mode.
    CrossHAR uses masked suffix; other modes (including CRT) use default naming.
    """
    input_tag = get_imu_input_tag(input_channels)
    if mode == "CrossHAR":
        target = f"pretrain_{mode}_{input_tag}_masked_{lambda1}_{lambda2}"
    else:
        target = f"pretrain_{mode}_{input_tag}"

    embedding_path = os.path.join(embed_dir, f"embedding_{target}.npy")
    label_path = os.path.join(embed_dir, f"label_{target}.npy")
    return embedding_path, label_path, target


if __name__ == "__main__":
    training_rate = 0.8
    balance = True

    args = handle_classifier_argv(default_mode="BioBankSSL", default_classifier_prefix="transformer")
    label_rate = args.train_subset_rate
    mode = args.mode
    method = args.classifier_prefix

    lambda1 = 6.0
    lambda2 = 1.0

    embedding_path, label_path, target = resolve_embedding_and_label_paths(
        mode=mode,
        lambda1=lambda1,
        lambda2=lambda2,
        embed_dir=args.embed_dir,
        input_channels=args.input_channels,
    )

    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    print("Loading embedding from:", embedding_path)
    print("Loading labels from:", label_path)
    print("Train/validation split: 80% / 20%")

    embedding = np.load(embedding_path).astype(np.float32)
    labels = np.load(label_path).astype(np.float32)

    classify_embeddings(
        args=args,
        data=embedding,
        labels=labels,
        label_index=0,
        training_rate=training_rate,
        label_rate=label_rate,
        balance=balance,
        method=method
    )
