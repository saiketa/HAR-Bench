import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import test as test_lib
from models.CRT import CRT4Pretrain
from config import TrainConfig
from subject_cv_utils import (
    DEFAULT_DATASET_8_NAMES,
    flatten_label_column,
    get_dataset_names,
    get_fold_subject_split,
    load_dataset8_arrays,
)
from utils.preprocessors import IMUDataset, Preprocess4CRT
from utils.utils import get_device, get_imu_input_tag, set_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one cross-subject CV fold")
    parser.add_argument("--datasets_root", type=str, default="./data")
    parser.add_argument("-td", "--dataset_names", nargs="+", default=None)
    parser.add_argument("--data_config", type=str, default="./data/data_config.json")
    parser.add_argument("--mode", type=str, default="BioBankSSL",
                        choices=["LIMU-BERT", "TS-TCC", "TS2Vec", "SimMTM", "BioBankSSL", "FOCAL", "CrossHAR", "CRT"])
    parser.add_argument("--encoder_backbone", type=str, default="transformer",
                        choices=["transformer", "cnn", "resnet"])
    parser.add_argument("--classifier_prefix", type=str, default="transformer",
                        choices=["gru", "transformer", "mlp", "cnn"])
    parser.add_argument("--model_version", type=str, default="v1")
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subject_label_index", type=int, default=1)
    parser.add_argument("--pretrain_train_cfg", type=str, default=None)
    parser.add_argument("--classifier_train_cfg", type=str, default="./config/classifier.json")
    parser.add_argument("--save_dir", type=str, default="./save_subject_cv")
    parser.add_argument("--input_channels", type=int, default=6, choices=[3, 6])
    parser.add_argument("--lambda1", type=float, default=6.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    parser.add_argument("-g", "--gpu", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def classifier_output_dim(labels_list):
    labels = np.concatenate([flatten_label_column(x, 0) for x in labels_list], axis=0)
    return int(labels.max()) + 1


def main():
    args = parse_args()
    if args.pretrain_train_cfg is None:
        args.pretrain_train_cfg = f"./config/{args.mode}.json"
    classifier_train_cfg = TrainConfig.from_json(args.classifier_train_cfg)
    if args.seed is not None:
        classifier_train_cfg = classifier_train_cfg._replace(seed=int(args.seed))
    set_seeds(classifier_train_cfg.seed)

    args.pretrain_datasets = get_dataset_names(args.datasets_root, args.dataset_names or DEFAULT_DATASET_8_NAMES)
    dataset_names, data_list, label_list = load_dataset8_arrays(args, keep_all_label_dims=True)
    split_seed = classifier_train_cfg.seed
    _, test_subjects, _ = get_fold_subject_split(
        label_list=label_list,
        fold_id=args.fold_id,
        n_folds=args.n_folds,
        seed=split_seed,
        subject_label_index=args.subject_label_index,
    )
    print(f"Fold {args.fold_id}/{args.n_folds}: test subjects {test_subjects.tolist()}")

    device = get_device(args.gpu)
    batch_size = args.batch_size if args.batch_size is not None else classifier_train_cfg.batch_size
    input_tag = get_imu_input_tag(args.input_channels)
    fold_save_dir = os.path.join(args.save_dir, f"fold_{args.fold_id}")

    test_args = argparse.Namespace(
        mode=args.mode,
        encoder_backbone=args.encoder_backbone,
        classifier_prefix=args.classifier_prefix,
        model_version=args.model_version,
        input_channels=args.input_channels,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        subject_cv=False,
        cv_fold=args.fold_id,
    )
    pretrain_model_cfg, pretrain_model = test_lib.build_pretrain_model(test_args)
    if args.mode == "CRT":
        warm_dataset = IMUDataset(
            data_list[0],
            label_list[0],
            feature_len=pretrain_model_cfg.feature_num,
            pipeline=[Preprocess4CRT(feature_len=pretrain_model_cfg.feature_num, return_tensor=False)],
        )
        sample_len = int(warm_dataset[0][0].shape[0])
        if hasattr(pretrain_model_cfg, "_replace"):
            pretrain_model_cfg = pretrain_model_cfg._replace(seq_len=sample_len)
        else:
            from types import SimpleNamespace
            pretrain_model_cfg = SimpleNamespace(**vars(pretrain_model_cfg))
            pretrain_model_cfg.seq_len = sample_len
        pretrain_model = CRT4Pretrain(pretrain_model_cfg)
    if args.mode == "CrossHAR":
        pretrain_ckpt = os.path.join(
            fold_save_dir,
            f"pretrain_{args.mode}_{input_tag}_masked_{args.lambda1}_{args.lambda2}.pt",
        )
    else:
        pretrain_ckpt = os.path.join(fold_save_dir, f"pretrain_{args.mode}_{input_tag}.pt")
    classifier_ckpt = os.path.join(
        fold_save_dir,
        f"classifier_{args.mode}_{args.classifier_prefix}_{input_tag}.pt",
    )
    if not pretrain_ckpt or not os.path.exists(pretrain_ckpt):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_ckpt}")
    if not os.path.exists(classifier_ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_ckpt}")

    test_lib.load_pretrain_checkpoint(pretrain_model, pretrain_ckpt, device, args.mode)
    pretrain_model = pretrain_model.to(device)

    first_data = data_list[0]
    first_labels = label_list[0]
    warm_embeddings, _ = test_lib.extract_embeddings(
        pretrain_model=pretrain_model,
        data=first_data[: min(8, len(first_data))],
        labels=first_labels[: min(8, len(first_labels))],
        feature_len=pretrain_model_cfg.feature_num,
        batch_size=batch_size,
        device=device,
        mode=args.mode,
        dataset_name=dataset_names[0],
        seq_len=getattr(pretrain_model_cfg, "seq_len", first_data.shape[1]),
    )
    input_dim = warm_embeddings.shape[-1]
    seq_len = 1 if warm_embeddings.ndim == 2 else warm_embeddings.shape[1]
    label_num = classifier_output_dim(label_list)
    _, classifier_model = test_lib.build_classifier_model(
        test_args,
        input_dim=input_dim,
        output_dim=label_num,
        seq_len=seq_len,
    )
    test_lib.load_classifier_checkpoint(classifier_model, classifier_ckpt, device)
    classifier_model = classifier_model.to(device)

    results = []
    all_true = []
    all_pred = []
    for dataset_name, data, labels in zip(dataset_names, data_list, label_list):
        subjects = flatten_label_column(labels, args.subject_label_index)
        mask = np.isin(subjects, test_subjects)
        if not np.any(mask):
            continue
        data_fold = data[mask]
        labels_fold = labels[mask]
        embeddings, labels_flat = test_lib.extract_embeddings(
            pretrain_model=pretrain_model,
            data=data_fold,
            labels=labels_fold,
            feature_len=pretrain_model_cfg.feature_num,
            batch_size=batch_size,
            device=device,
            mode=args.mode,
            dataset_name=dataset_name,
            seq_len=getattr(pretrain_model_cfg, "seq_len", data_fold.shape[1]),
        )
        acc, f1, y_true, y_pred = test_lib.evaluate_classifier(
            classifier_model=classifier_model,
            embeddings=embeddings,
            labels=labels_flat,
            batch_size=batch_size,
            device=device,
        )
        print(f"[{dataset_name}] samples={len(labels_flat)} Acc={acc:.4f}, Macro-F1={f1:.4f}")
        results.append({"dataset": dataset_name, "num_samples": len(labels_flat), "acc": acc, "macro_f1": f1})
        all_true.append(y_true)
        all_pred.append(y_pred)

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    overall = {
        "dataset": "ALL_DATASETS",
        "num_samples": len(all_true),
        "acc": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
    }
    df = pd.concat([pd.DataFrame(results), pd.DataFrame([overall])], ignore_index=True)
    print("\n=== Summary ===")
    print(df)
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
