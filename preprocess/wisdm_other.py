#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

from wisdm import (
    BODY_PART_BY_POSITION,
    DATASET_ID,
    DATASET_PATH,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    SEQ_LEN,
    SR,
    VERSION,
    collect_sensor_files,
    ensure_dir,
    parse_wisdm_file,
)


SAVE_DIR = os.path.join('dataset_other', 'WISDM')
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

# WISDM other raw activity label -> unified benchmark label
WISDM_OTHER_TO_BENCH = {
    "F": 32,   # Typing
    "G": 38,   # Brushing Teeth
    "H": 39,   # Eating Soup
    "I": 40,   # Eating Chips
    "J": 41,   # Eating Pasta
    "K": 42,   # Drinking from Cup
    "L": 43,   # Eating Sandwich
    "M": 212,  # Kicking (Soccer Ball)
    "O": 213,  # Tennis Ball
    "P": 207,  # Basketball
    "Q": 33,   # Writing
    "R": 44,   # Clapping
    "S": 25,   # Folding Clothes
}


def build_aligned_windows_other(acc_df, gyro_df, seq_len):
    """
    Align acc and gyro by exact (user, activity, timestamp), then split
    by user/activity and make non-overlapping windows.
    """
    if len(acc_df) == 0 or len(gyro_df) == 0:
        return []

    merged = pd.merge(
        acc_df,
        gyro_df,
        on=["user", "activity", "timestamp"],
        suffixes=("_acc", "_gyro")
    )

    if len(merged) == 0:
        return []

    merged = merged.sort_values(["user", "activity", "timestamp"]).reset_index(drop=True)
    merged = merged[merged["activity"].isin(WISDM_OTHER_TO_BENCH.keys())].reset_index(drop=True)
    if len(merged) == 0:
        return []

    results = []
    start = 0
    n_rows = len(merged)

    def flush_segment(seg_df):
        if len(seg_df) < seq_len:
            return

        sensor = seg_df[["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]].to_numpy(dtype=np.float32)
        user_id = int(seg_df.iloc[0]["user"]) - 1600
        raw_act = seg_df.iloc[0]["activity"]
        bench_activity = WISDM_OTHER_TO_BENCH[raw_act]

        n_win = sensor.shape[0] // seq_len
        if n_win == 0:
            return

        sensor = sensor[:n_win * seq_len]
        sensor = sensor.reshape(n_win, seq_len, 6)

        for i in range(n_win):
            results.append((sensor[i], user_id, bench_activity))

    for i in range(1, n_rows):
        prev = merged.iloc[i - 1]
        cur = merged.iloc[i]

        same_stream = (
            prev["user"] == cur["user"] and
            prev["activity"] == cur["activity"]
        )

        if not same_stream:
            flush_segment(merged.iloc[start:i])
            start = i

    flush_segment(merged.iloc[start:n_rows])

    return results


def process_wisdm_other(dataset_path, out_root, version):
    data_list = []
    label_list = []

    for position_name, body_part_id in BODY_PART_BY_POSITION.items():
        acc_dir = os.path.join(dataset_path, position_name, "accel")
        gyro_dir = os.path.join(dataset_path, position_name, "gyro")

        if not os.path.isdir(acc_dir):
            raise FileNotFoundError(f"Missing directory: {acc_dir}")
        if not os.path.isdir(gyro_dir):
            raise FileNotFoundError(f"Missing directory: {gyro_dir}")

        acc_files = collect_sensor_files(acc_dir, "accel")
        gyro_files = collect_sensor_files(gyro_dir, "gyro")

        common_ids = sorted(set(acc_files.keys()) & set(gyro_files.keys()))
        print(f"[{position_name}] matched file pairs: {len(common_ids)}")

        for file_id in common_ids:
            acc_df = parse_wisdm_file(acc_files[file_id])
            gyro_df = parse_wisdm_file(gyro_files[file_id])

            windows = build_aligned_windows_other(acc_df, gyro_df, SEQ_LEN)

            for samp, user_id, bench_activity in windows:
                global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id
                data_list.append(samp)
                label_list.append([[bench_activity, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No valid samples produced. Check WISDM path / folder names / file format.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    label = np.array(label_list, dtype=np.int64)

    print(f"All data processed. Size: {data.shape[0]}")
    print("Data shape:", data.shape)
    print("Label shape:", label.shape)
    print("Activity classes contained:", np.unique(label[:, 0, 0]).astype(int).tolist())
    print("Global subject ids contained:", np.unique(label[:, 0, 1]).astype(int).tolist())
    print("Body part ids contained:", np.unique(label[:, 0, 2]).astype(int).tolist())
    print("Dataset ids contained:", np.unique(label[:, 0, 3]).astype(int).tolist())

    ensure_dir(out_root)
    np.save(os.path.join(out_root, "data.npy"), data)
    np.save(os.path.join(out_root, "label.npy"), label)

    return data, label


def update_data_config(config_path, version, data, label):
    dataset_name = f"WISDM_OTHER_{version}"

    size = int(data.shape[0])

    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            print("[Warning] data_config.json is invalid JSON. Recreating it.")

    cfg[dataset_name] = {
        "sr": SR,
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

    print(f"Updated {config_path} with key: {dataset_name}")


if __name__ == "__main__":
    data, label = process_wisdm_other(
        dataset_path=DATASET_PATH,
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
