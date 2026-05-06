#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import pandas as pd


DATASET_PATH = r'WISDM'
SAVE_DIR = os.path.join('dataset_8', 'WISDM')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

SR = 20
SEQ_LEN = 120
DIMENSION = 6
GLOBAL_SUBJECT_ID_OFFSET = 282
DATASET_ID = 13

BENCH_ACTIVITY_LABEL = [
    "sitting",
    "standing",
    "lying",
    "upstairs",
    "downstairs",
    "walking",
    "running",
    "jumping"
]

# WISDM label -> unified benchmark label
WISDM_TO_BENCH = {
    "A": 5,   # walking
    "B": 6,   # jogging -> running
    "C": 3,   # stairs -> upstairs
    "D": 0,   # sitting
    "E": 1,   # standing
}

BODY_PART_BY_POSITION = {
    "phone": 7,
    "watch": 1,
}

EXPECTED_SENSOR_COLS = ["user", "activity", "timestamp", "x", "y", "z"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_wisdm_file(path):
    """
    Parse WISDM txt file into DataFrame with columns:
    user, activity, timestamp, x, y, z

    Handles lines ending with ';'
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # remove trailing semicolon
            if line.endswith(";"):
                line = line[:-1]

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue

            try:
                user = int(parts[0])
                activity = parts[1]
                timestamp = int(float(parts[2]))
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except Exception:
                continue

            rows.append([user, activity, timestamp, x, y, z])

    if len(rows) == 0:
        return pd.DataFrame(columns=EXPECTED_SENSOR_COLS)

    df = pd.DataFrame(rows, columns=EXPECTED_SENSOR_COLS)
    return df


def extract_file_id(filename):
    """
    Extract numeric id from:
    data_1601_accel_phone.txt
    data_1601_gyro_phone.txt
    """
    m = re.search(r"data_(\d+)_", filename)
    if m is None:
        return None
    return int(m.group(1))


def collect_sensor_files(sensor_dir, sensor_keyword):
    """
    Return dict[file_id] = file_path
    """
    result = {}
    for fname in sorted(os.listdir(sensor_dir)):
        if not fname.endswith(".txt"):
            continue
        if sensor_keyword not in fname:
            continue
        file_id = extract_file_id(fname)
        if file_id is None:
            continue
        result[file_id] = os.path.join(sensor_dir, fname)
    return result


def build_aligned_windows(acc_df, gyro_df, seq_len):
    """
    Align acc and gyro by exact (user, activity, timestamp), then
    split by contiguous sequence and make non-overlapping windows.

    Return list of tuples:
    (window_data [seq_len, 6], user_id, bench_activity)
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

    # keep only supported activities
    merged = merged[merged["activity"].isin(WISDM_TO_BENCH.keys())].reset_index(drop=True)
    if len(merged) == 0:
        return []

    results = []

    start = 0
    N = len(merged)

    def flush_segment(seg_df):
        if len(seg_df) < seq_len:
            return

        sensor = seg_df[["x_acc", "y_acc", "z_acc", "x_gyro", "y_gyro", "z_gyro"]].to_numpy(dtype=np.float32)
        user_id = int(seg_df.iloc[0]["user"]) - 1600   # 1600~1650 -> 0~50
        raw_act = seg_df.iloc[0]["activity"]
        bench_activity = WISDM_TO_BENCH[raw_act]

        n_win = sensor.shape[0] // seq_len
        if n_win == 0:
            return

        sensor = sensor[:n_win * seq_len]
        sensor = sensor.reshape(n_win, seq_len, 6)

        for i in range(n_win):
            results.append((sensor[i], user_id, bench_activity))

    for i in range(1, N):
        prev = merged.iloc[i - 1]
        cur = merged.iloc[i]

        same_stream = (
            prev["user"] == cur["user"] and
            prev["activity"] == cur["activity"]
        )

        # timestamp not strictly continuous requirement;
        # only break when user/activity changes
        if not same_stream:
            flush_segment(merged.iloc[start:i])
            start = i

    flush_segment(merged.iloc[start:N])

    return results


def process_wisdm(dataset_path, out_root, version):
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

            windows = build_aligned_windows(acc_df, gyro_df, SEQ_LEN)

            for samp, user_id, bench_activity in windows:
                global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id
                data_list.append(samp)
                label_list.append([[bench_activity, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No valid samples produced. Check WISDM path / folder names / file format.")

    data = np.stack(data_list, axis=0).astype(np.float32)   # [N, 120, 6]
    label = np.array(label_list, dtype=np.int64)            # [N, 1, 4]

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
    dataset_name = f"WISDM_{version}"

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
    data, label = process_wisdm(
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
