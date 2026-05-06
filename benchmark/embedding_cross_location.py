import argparse
import os

import embedding
import numpy as np
from cross_location_utils import (
    add_common_cross_location_args,
    build_cross_location_pretrain_args,
    load_dataset_arrays,
    pretrain_ckpt_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate embeddings for one cross-location CV fold")
    add_common_cross_location_args(parser)
    parser.add_argument("--train_cfg", type=str, default=None)
    parser.add_argument("--lambda1", type=float, default=6.0)
    parser.add_argument("--lambda2", type=float, default=1.0)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    run_args = build_cross_location_pretrain_args(cli_args, train_cfg_path=cli_args.train_cfg)
    run_args.seed = cli_args.seed

    ckpt_name = pretrain_ckpt_name(cli_args.mode, cli_args.input_channels, cli_args.lambda1, cli_args.lambda2)
    ckpt_file = os.path.join(run_args.save_dir, ckpt_name + ".pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_file}")

    dataset_names, data_list, label_list = load_dataset_arrays(run_args)
    run_args.pretrain_datasets = dataset_names

    def load_cross_location_embedding_data(datasets_root, dataset_names, dataset_version="20_120"):
        return data_list, label_list

    embedding.load_multiple_pretrain_datasets = load_cross_location_embedding_data
    embedding.generate_embedding_or_output(
        args=run_args,
        output_embed=True,
        save=True,
        lambda1=cli_args.lambda1,
        lambda2=cli_args.lambda2,
    )

    full_labels = np.concatenate(label_list, axis=0).astype(np.float32)
    dataset_ids = np.concatenate(
        [np.full((labels.shape[0],), i, dtype=np.int64) for i, labels in enumerate(label_list)],
        axis=0,
    )
    label_path = os.path.join(run_args.embed_dir, f"label_{ckpt_name}.npy")
    dataset_id_path = os.path.join(run_args.embed_dir, f"dataset_id_{ckpt_name}.npy")
    np.save(label_path, full_labels)
    np.save(dataset_id_path, dataset_ids)
    print(f"Saved cross-location full labels to: {label_path}")


if __name__ == "__main__":
    main()
