import argparse
import os

import numpy as np
import pandas as pd

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
)
from models.classifiers import fetch_classifier
from test import evaluate_classifier, load_classifier_checkpoint
from utils.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Test one cross-location CV fold")
    add_common_cross_location_args(parser)
    parser.add_argument("--classifier_prefix", type=str, default="cnn", choices=["gru", "transformer", "mlp", "cnn"])
    parser.add_argument("--classifier_train_cfg", type=str, default="./config/classifier.json")
    parser.add_argument("--lambda1", type=float, default=6.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_location = canonical_location_name(args.train_location)
    test_location = canonical_location_name(args.test_location)
    if train_location == test_location:
        raise ValueError("train_location and test_location must be different for cross-location validation.")

    run_args = build_cross_location_pretrain_args(args)
    ckpt_name = pretrain_ckpt_name(args.mode, args.input_channels, args.lambda1, args.lambda2)
    embedding_path = os.path.join(run_args.embed_dir, f"embedding_{ckpt_name}.npy")
    label_path = os.path.join(run_args.embed_dir, f"label_{ckpt_name}.npy")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    classifier_ckpt = os.path.join(
        run_args.save_dir,
        classifier_ckpt_name(args.mode, args.classifier_prefix, args.input_channels) + ".pt",
    )
    if not os.path.exists(classifier_ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_ckpt}")

    train_cfg = TrainConfig.from_json(args.classifier_train_cfg)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.batch_size
    device = get_device(args.gpu)

    embeddings = np.load(embedding_path).astype(np.float32)
    labels_full = np.load(label_path).astype(np.float32)
    if embeddings.ndim == 2:
        embeddings = embeddings[:, None, :]
    if labels_full.ndim == 2:
        labels_full = labels_full[:, None, :]

    _, _, label_list = load_dataset_arrays(run_args)
    _, test_subjects, _ = get_fold_subject_split(
        label_list,
        fold_id=args.fold_id,
        n_folds=args.n_folds,
        subject_label_index=args.subject_label_index,
    )

    labels = flatten_label_column(labels_full, args.activity_label_index).astype(np.int64)
    test_mask = mask_by_subjects_and_location(
        labels_full,
        test_subjects,
        test_location,
        subject_label_index=args.subject_label_index,
        location_label_index=args.location_label_index,
    )
    if not np.any(test_mask):
        raise ValueError(f"No test samples for test_location={test_location}, fold={args.fold_id}")

    model_cfg = load_model_config(
        f"classifier_{args.mode}_{args.classifier_prefix}",
        args.classifier_prefix,
        args.model_version,
    )
    if model_cfg is None:
        raise ValueError(f"Unable to load classifier config: {args.classifier_prefix}")
    if hasattr(model_cfg, "_replace"):
        model_cfg = model_cfg._replace(seq_len=int(embeddings.shape[1]))
    else:
        from types import SimpleNamespace

        model_cfg = SimpleNamespace(**vars(model_cfg))
        model_cfg.seq_len = int(embeddings.shape[1])

    label_num = int(np.max(labels)) + 1
    classifier_model = fetch_classifier(
        args.classifier_prefix,
        model_cfg,
        input=embeddings.shape[-1],
        output=label_num,
    )
    load_classifier_checkpoint(classifier_model, classifier_ckpt, device)
    classifier_model = classifier_model.to(device)

    acc, f1, _, _ = evaluate_classifier(
        classifier_model=classifier_model,
        embeddings=embeddings[test_mask],
        labels=labels[test_mask],
        batch_size=batch_size,
        device=device,
    )
    print(
        f"[fold={args.fold_id} train={train_location} -> test={test_location}] "
        f"samples={int(test_mask.sum())} Acc={acc:.4f}, Macro-F1={f1:.4f}"
    )

    row = {
        "fold": int(args.fold_id),
        "train_location": train_location,
        "test_location": test_location,
        "num_samples": int(test_mask.sum()),
        "acc": acc,
        "macro_f1": f1,
    }
    df = pd.DataFrame([row])
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        write_header = not os.path.exists(args.output_csv)
        df.to_csv(args.output_csv, mode="a", header=write_header, index=False)
        print(f"Saved result to: {args.output_csv}")
    else:
        print(df)


if __name__ == "__main__":
    main()
