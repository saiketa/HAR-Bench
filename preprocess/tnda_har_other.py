#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from tnda_har import (
    BODY_PART_IDS,
    DATASET_ID,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    SEQ_LEN,
    TARGET_SR,
    WIN_SRC,
    WIN_TGT,
    ensure_dir,
    fill_nan_linear,
    iter_full_windows,
    resample_linear,
    robust_read_csv_46,
    split_45_to_5x6,
    subject_id_from_name,
)


DATASET_PATH = r'TNDA-HAR'
SAVE_DIR = os.path.join('dataset_other', 'TNDA-HAR')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

# TNDA-HAR other raw activity id -> unified benchmark activity id
RAW_ACTIVITY_TO_BENCH = {
    6: 19,   # cycling
}

TARGET_IDS = sorted(RAW_ACTIVITY_TO_BENCH.keys())


def process_tnda_har_other(tnda_root, out_root, version):
    data_list = []
    label_list = []

    for fname in sorted(os.listdir(tnda_root)):
        if not fname.lower().endswith(".csv"):
            continue

        subj_path = os.path.join(tnda_root, fname)
        user_id = subject_id_from_name(os.path.splitext(fname)[0])
        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

        arr = robust_read_csv_46(subj_path)

        if np.isnan(arr).any():
            arr[:, :45] = fill_nan_linear(arr[:, :45])

        act_col = arr[:, 45].astype(np.int32)

        keep_mask = np.isin(act_col, TARGET_IDS)
        if not keep_mask.any():
            continue

        arr = arr[keep_mask, :]
        act_col = act_col[keep_mask]

        for raw_id in TARGET_IDS:
            seg = arr[act_col == raw_id, :]
            if seg.shape[0] < WIN_SRC:
                continue

            data_45 = seg[:, :45]

            for win in iter_full_windows(data_45, WIN_SRC):
                win = fill_nan_linear(win)
                win120 = resample_linear(win, WIN_TGT)

                samples = split_45_to_5x6(win120)
                for pos_id, samp in enumerate(samples):
                    bench_act = RAW_ACTIVITY_TO_BENCH[raw_id]
                    body_part_id = BODY_PART_IDS[pos_id]

                    data_list.append(samp)
                    label_list.append([[bench_act, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check TNDA-HAR path / CSV format / filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    label = np.array(label_list, dtype=np.int64)

    print(f'All data processed. Size: {data.shape[0]}')
    print('Data shape:', data.shape)
    print('Label shape:', label.shape)
    print('Activity classes contained:', np.unique(label[:, 0, 0]).astype(int).tolist())
    print('Global subject ids contained:', np.unique(label[:, 0, 1]).astype(int).tolist())
    print('Body part ids contained:', np.unique(label[:, 0, 2]).astype(int).tolist())
    print('Dataset ids contained:', np.unique(label[:, 0, 3]).astype(int).tolist())

    ensure_dir(out_root)
    np.save(os.path.join(out_root, 'data.npy'), data)
    np.save(os.path.join(out_root, 'label.npy'), label)

    return data, label


def update_data_config(config_path, version, data, label):
    dataset_name = f'TNDA-HAR_OTHER_{version}'

    size = int(data.shape[0])

    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            print("[Warning] data_config.json is invalid JSON. Recreating it.")

    cfg[dataset_name] = {
        "sr": TARGET_SR,
        "seq_len": SEQ_LEN,
        "dimension": DIMENSION,
        "activity_label_index": 0,
        "global_subject_label_index": 1,
        "body_part_label_index": 2,
        "dataset_label_index": 3,
        "size": size
    }

    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f'Updated {config_path} with key: {dataset_name}')


if __name__ == "__main__":
    data, label = process_tnda_har_other(
        tnda_root=DATASET_PATH,
        out_root=SAVE_DIR,
        version=VERSION
    )

    update_data_config(
        config_path=CONFIG_PATH,
        version=VERSION,
        data=data,
        label=label
    )

    print("[DONE]")
    print(f"Saved to: {SAVE_DIR}")
    print(" - data.npy")
    print(" - label.npy")
    print(f"Config path: {CONFIG_PATH}")
