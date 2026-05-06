import argparse

import pretrain
from subject_cv_utils import (
    add_common_subject_cv_args,
    build_pretrain_args,
    filter_by_subjects,
    get_fold_subject_split,
    load_dataset8_arrays,
    load_train_and_mask_cfg,
    print_fold_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-subject CV pretraining on data/")
    add_common_subject_cv_args(parser)
    parser.add_argument("--training_rate", type=float, default=0.8,
                        help="Sample-level train rate inside the training-subject pool; validation uses 0.2")
    parser.add_argument("--train_cfg", type=str, default=None,
                        help="Override pretrain config path")
    parser.add_argument("--subject_label_index", type=int, default=1)
    return parser.parse_args()


def main():
    cli_args = parse_args()
    run_args = build_pretrain_args(cli_args, train_cfg_path=cli_args.train_cfg)
    run_args.seed = cli_args.seed

    train_cfg, mask_cfg = load_train_and_mask_cfg(run_args)
    _, data_list, label_list = load_dataset8_arrays(run_args, keep_all_label_dims=True)

    split_seed = train_cfg.seed if cli_args.seed is None else cli_args.seed
    train_subjects, test_subjects, _ = get_fold_subject_split(
        label_list=label_list,
        fold_id=cli_args.fold_id,
        n_folds=cli_args.n_folds,
        seed=split_seed,
        subject_label_index=cli_args.subject_label_index,
    )
    print_fold_summary(
        cli_args.fold_id,
        train_subjects,
        test_subjects,
        data_list,
        label_list,
        subject_label_index=cli_args.subject_label_index,
    )

    train_data_list, train_label_list = filter_by_subjects(
        data_list,
        label_list,
        train_subjects,
        subject_label_index=cli_args.subject_label_index,
    )
    usable = []
    skipped = []
    for name, data, labels in zip(run_args.pretrain_datasets, train_data_list, train_label_list):
        train_num = int(data.shape[0] * cli_args.training_rate)
        vali_num = int(data.shape[0] * 0.1)
        if data.shape[0] == 0 or train_num == 0 or vali_num == 0:
            skipped.append((name, int(data.shape[0]), train_num, vali_num))
            continue
        usable.append((name, data, labels))

    if skipped:
        print(f"Skipped tiny/empty training datasets for fold {cli_args.fold_id}: {skipped}")
    if not usable:
        raise ValueError(f"No training samples remain after subject split for fold {cli_args.fold_id}")

    run_args.pretrain_datasets = [name for name, _, _ in usable]
    train_data_list = [data for _, data, _ in usable]
    train_label_list = [labels for _, _, labels in usable]

    # Keep subject-CV independent from pretrain.py edits.  The original
    # pretrain.main() reads data through its module-level
    # load_multi_pretrain_data_config symbol, so this process-local injection
    # lets us reuse the unchanged training implementation with data that has
    # already had the held-out subjects removed.
    def load_subject_cv_data_config(_args):
        return train_data_list, train_label_list, train_cfg, run_args.model_cfg, mask_cfg

    pretrain.load_multi_pretrain_data_config = load_subject_cv_data_config
    if hasattr(run_args, "subject_cv"):
        run_args.subject_cv = False

    pretrain.main(
        args=run_args,
        training_rate=cli_args.training_rate,
        mode=cli_args.mode,
    )


if __name__ == "__main__":
    main()
