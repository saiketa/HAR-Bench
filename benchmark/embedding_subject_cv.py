import argparse
import os

import embedding
import numpy as np
from utils.utils import get_imu_input_tag
from subject_cv_utils import (
    add_common_subject_cv_args,
    build_pretrain_args,
    load_dataset8_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for one cross-subject CV fold")
    add_common_subject_cv_args(parser)
    parser.add_argument("--train_cfg", type=str, default=None, help="Override pretrain config path")
    parser.add_argument("--lambda1", type=float, default=6.0, help="CrossHAR loss weight lambda1")
    parser.add_argument("--lambda2", type=float, default=1.0, help="CrossHAR loss weight lambda2")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    run_args = build_pretrain_args(cli_args, train_cfg_path=cli_args.train_cfg)
    run_args.seed = cli_args.seed
    run_args.subject_cv = True
    run_args.cv_fold = cli_args.fold_id
    run_args.cv_folds = cli_args.n_folds
    run_args.subject_label_index = 1

    if cli_args.mode == "CrossHAR":
        ckpt_file = f"{run_args.save_path}_masked_{cli_args.lambda1}_{cli_args.lambda2}.pt"
    else:
        ckpt_file = run_args.save_path + ".pt"
    if not ckpt_file or not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_file}")

    dataset_names, data_list, label_list = load_dataset8_arrays(run_args, keep_all_label_dims=True)
    run_args.pretrain_datasets = dataset_names

    # Keep this wrapper independent from embedding.py's on-disk dataset naming.
    def load_subject_cv_embedding_data(datasets_root, dataset_names, dataset_version="20_120"):
        return data_list, label_list

    embedding.load_multiple_pretrain_datasets = load_subject_cv_embedding_data

    embedding.generate_embedding_or_output(
        args=run_args,
        output_embed=True,
        save=True,
        lambda1=cli_args.lambda1,
        lambda2=cli_args.lambda2,
    )

    # Subject-CV needs activity, subject, and an internal dataset id.
    input_tag = get_imu_input_tag(cli_args.input_channels)
    ckpt_name = f"pretrain_{cli_args.mode}_{input_tag}"
    if cli_args.mode == "CrossHAR":
        ckpt_name = f"{ckpt_name}_masked_{cli_args.lambda1}_{cli_args.lambda2}"

    full_labels = np.concatenate(label_list, axis=0).astype(np.float32)
    dataset_ids = np.concatenate(
        [np.full((labels.shape[0],), i, dtype=np.int64) for i, labels in enumerate(label_list)],
        axis=0,
    )
    if full_labels.ndim == 2:
        full_labels = full_labels[:, None, :]
    if full_labels.ndim != 3:
        raise ValueError(f"Unexpected full label shape: {full_labels.shape}")
    if full_labels.shape[-1] <= 2:
        dataset_label = np.broadcast_to(
            dataset_ids[:, None, None].astype(np.float32),
            (full_labels.shape[0], full_labels.shape[1], 1),
        )
        full_labels = np.concatenate([full_labels, dataset_label], axis=-1)

    label_save_path = os.path.join(run_args.embed_dir, f"label_{ckpt_name}.npy")
    dataset_id_save_path = os.path.join(run_args.embed_dir, f"dataset_id_{ckpt_name}.npy")
    np.save(label_save_path, full_labels)
    np.save(dataset_id_save_path, dataset_ids)
    print(f"Overwrote subject-CV full labels to: {label_save_path}")


if __name__ == "__main__":
    main()
