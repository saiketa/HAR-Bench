import argparse

import pretrain
from cross_location_utils import (
    add_common_cross_location_args,
    build_cross_location_pretrain_args,
    canonical_location_name,
    filter_by_subjects_and_location,
    filter_usable_datasets,
    get_fold_subject_split,
    load_dataset_arrays,
    load_train_and_mask_cfg,
    print_location_fold_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-location CV pretraining")
    add_common_cross_location_args(parser)
    parser.add_argument("--training_rate", type=float, default=0.8)
    parser.add_argument("--train_cfg", type=str, default=None)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    train_location = canonical_location_name(cli_args.train_location)
    test_location = canonical_location_name(cli_args.test_location)
    if train_location == test_location:
        raise ValueError("train_location and test_location must be different for cross-location validation.")

    run_args = build_cross_location_pretrain_args(cli_args, train_cfg_path=cli_args.train_cfg)
    run_args.seed = cli_args.seed

    train_cfg, mask_cfg = load_train_and_mask_cfg(run_args)
    dataset_names, data_list, label_list = load_dataset_arrays(run_args)
    split_seed = train_cfg.seed if cli_args.seed is None else cli_args.seed
    _ = split_seed  # The split is deterministic by sorted global subject ids.
    train_subjects, test_subjects, _ = get_fold_subject_split(
        label_list,
        fold_id=cli_args.fold_id,
        n_folds=cli_args.n_folds,
        subject_label_index=cli_args.subject_label_index,
    )
    print_location_fold_summary(
        cli_args.fold_id,
        train_location,
        test_location,
        train_subjects,
        test_subjects,
        data_list,
        label_list,
        subject_label_index=cli_args.subject_label_index,
        location_label_index=cli_args.location_label_index,
    )

    train_data_list, train_label_list = filter_by_subjects_and_location(
        data_list,
        label_list,
        train_subjects,
        train_location,
        subject_label_index=cli_args.subject_label_index,
        location_label_index=cli_args.location_label_index,
    )
    usable, skipped = filter_usable_datasets(
        dataset_names,
        train_data_list,
        train_label_list,
        training_rate=cli_args.training_rate,
    )
    if skipped:
        print(f"Skipped tiny/empty datasets for train_location={train_location}: {skipped}")
    if not usable:
        raise ValueError(f"No samples remain for train_location={train_location}, fold={cli_args.fold_id}")

    run_args.pretrain_datasets = [name for name, _, _ in usable]
    train_data_list = [data for _, data, _ in usable]
    train_label_list = [labels for _, _, labels in usable]
    print(f"Pretrain samples: {sum(x.shape[0] for x in train_data_list)}")

    def load_cross_location_data_config(_args):
        return train_data_list, train_label_list, train_cfg, run_args.model_cfg, mask_cfg

    pretrain.load_multi_pretrain_data_config = load_cross_location_data_config
    pretrain.main(
        args=run_args,
        training_rate=cli_args.training_rate,
        mode=cli_args.mode,
    )


if __name__ == "__main__":
    main()
