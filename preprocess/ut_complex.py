#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd


DATASET_PATH = r'UT-Complex'
SAVE_DIR = os.path.join('dataset_8', 'UT-Complex')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

SRC_SR = 50
TARGET_SR = 20
WIN_SEC = 6
WIN_SRC = SRC_SR * WIN_SEC      # 300
WIN_TGT = TARGET_SR * WIN_SEC   # 120

DIMENSION = 6
SEQ_LEN = WIN_TGT
GLOBAL_SUBJECT_ID = 281
DATASET_ID = 12

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

# UT-Complex raw label -> unified benchmark label
UT_COMPLEX_TO_BENCH = {
    11111: 5,  # walk -> walking
    11112: 1,  # stand -> standing
    11113: 6,  # jog -> running
    11114: 0,  # sit -> sitting
    11116: 3,  # upstairs -> upstairs
    11117: 4,  # downstairs -> downstairs
}

BODY_PART_FILE_MAP = {
    "smartphoneatpocket.csv": 7,
    "smartphoneatwrist.csv": 1,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fill_nan_linear(x):
    x = x.copy()
    L, D = x.shape
    idx = np.arange(L, dtype=np.float32)

    for d in range(D):
        col = x[:, d]
        mask = np.isnan(col)
        if not mask.any():
            continue
        if mask.all():
            x[:, d] = 0.0
            continue
        x[mask, d] = np.interp(idx[mask], idx[~mask], col[~mask])

    return x


def resample_linear(win_src, tgt_len):
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def iter_continuous_label_segments(df, label_col):
    labels = df[label_col].to_numpy()
    if len(labels) == 0:
        return

    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield int(labels[start]), df.iloc[start:i]
            start = i
    yield int(labels[start]), df.iloc[start:]


def iter_full_windows(arr, win_len):
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[:n * win_len]
    for i in range(n):
        yield trunc[i * win_len:(i + 1) * win_len]


def load_ut_complex_csv(file_path):
    """
    Robust loader for UT-Complex.
    Supports both:
    1) no-header numeric csv
    2) header csv

    Expected logical columns:
    0  timestamp
    1:3 accelerometer xyz
    4:6 linear acceleration xyz
    7:9 gyroscope xyz
    10:12 magnetometer xyz
    13 label
    """
    # 先尝试无表头读取
    df = pd.read_csv(file_path, header=None)

    # 如果第一行其实是表头字符串，会导致很多非数值；此时重新读一次
    first_row_numeric = pd.to_numeric(df.iloc[0], errors='coerce')
    if first_row_numeric.isna().sum() > len(df.columns) // 2:
        df = pd.read_csv(file_path)

    return df


def extract_sensor_and_label(df):
    """
    Return:
        sensor: ndarray [T, 6] = linear_acc(3) + gyro(3)
        label: ndarray [T]
    """
    # 无表头情况：直接按列索引
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        if df.shape[1] < 14:
            raise ValueError(f'Expected at least 14 columns, got {df.shape[1]}')

        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        sensor = numeric_df.iloc[:, [4, 5, 6, 7, 8, 9]].to_numpy(dtype=np.float32)
        label = numeric_df.iloc[:, 13].to_numpy()
        return sensor, label

    # 有表头情况：尽量按列名匹配
    cols = [str(c).strip().lower().replace(" ", "").replace("-", "").replace("_", "") for c in df.columns]
    col_map = {cols[i]: df.columns[i] for i in range(len(cols))}

    linacc_candidates = [
        ["linearaccelerationsensorx", "linearaccelerationsensory", "linearaccelerationsensorz"],
        ["linearaccelerationx", "linearaccelerationy", "linearaccelerationz"],
        ["linaccx", "linaccy", "linaccz"],
    ]
    gyro_candidates = [
        ["gyroscopex", "gyroscopey", "gyroscopez"],
        ["gyrox", "gyroy", "gyroz"],
    ]
    label_candidates = ["activitylabel", "label", "activity"]

    linacc_cols = None
    gyro_cols = None
    label_col = None

    for cand in linacc_candidates:
        if all(c in col_map for c in cand):
            linacc_cols = [col_map[c] for c in cand]
            break

    for cand in gyro_candidates:
        if all(c in col_map for c in cand):
            gyro_cols = [col_map[c] for c in cand]
            break

    for c in label_candidates:
        if c in col_map:
            label_col = col_map[c]
            break

    if linacc_cols is None or gyro_cols is None or label_col is None:
        # fallback 到固定列位置
        if df.shape[1] < 14:
            raise ValueError(f'Cannot resolve columns and shape is only {df.shape}')
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        sensor = numeric_df.iloc[:, [4, 5, 6, 7, 8, 9]].to_numpy(dtype=np.float32)
        label = numeric_df.iloc[:, 13].to_numpy()
        return sensor, label

    numeric_df = df.copy()
    for c in linacc_cols + gyro_cols + [label_col]:
        numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce')

    sensor = numeric_df[linacc_cols + gyro_cols].to_numpy(dtype=np.float32)
    label = numeric_df[label_col].to_numpy()
    return sensor, label


def process_ut_complex(dataset_path, out_root, version):
    data_list = []
    label_list = []

    for file_name, body_part_id in BODY_PART_FILE_MAP.items():
        file_path = os.path.join(dataset_path, file_name)
        if not os.path.isfile(file_path):
            print(f'[Warning] Missing file: {file_path}')
            continue

        df = load_ut_complex_csv(file_path)
        sensor, label_arr = extract_sensor_and_label(df)

        valid_mask = ~np.isnan(label_arr)
        sensor = sensor[valid_mask]
        label_arr = label_arr[valid_mask].astype(np.int64)

        keep_mask = np.isin(label_arr, list(UT_COMPLEX_TO_BENCH.keys()))
        sensor = sensor[keep_mask]
        label_arr = label_arr[keep_mask]

        if len(label_arr) == 0:
            continue

        tmp_df = pd.DataFrame(sensor, columns=["ax", "ay", "az", "gx", "gy", "gz"])
        tmp_df["label"] = label_arr

        for raw_activity, seg_df in iter_continuous_label_segments(tmp_df, "label"):
            if raw_activity not in UT_COMPLEX_TO_BENCH:
                continue

            if len(seg_df) < WIN_SRC:
                continue

            bench_activity = UT_COMPLEX_TO_BENCH[raw_activity]
            seg_sensor = seg_df[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
            seg_sensor = fill_nan_linear(seg_sensor)

            for win in iter_full_windows(seg_sensor, WIN_SRC):
                samp = resample_linear(win, WIN_TGT)
                data_list.append(samp.astype(np.float32))

                label_list.append([[bench_activity, GLOBAL_SUBJECT_ID, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError('No valid samples produced. Check UT-Complex path / csv format / labels.')

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
    dataset_name = f'UT-COMPLEX_{version}'

    size = int(data.shape[0])

    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            print('[Warning] data_config.json is invalid JSON. Recreating it.')

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

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f'Updated {config_path} with key: {dataset_name}')


if __name__ == '__main__':
    data, label = process_ut_complex(
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

    print('[DONE]')
    print(f'Saved to: {SAVE_DIR}')
    print(' - data.npy')
    print(' - label.npy')
    print(f'Config path: {CONFIG_PATH}')
