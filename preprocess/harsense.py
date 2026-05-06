#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import pandas as pd


DATASET_PATH = r'HARSense'
SAVE_DIR = os.path.join('dataset_8', 'HARSense')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

SRC_SR = 50
TARGET_SR = 20
WIN_SEC = 6
WIN_SRC = SRC_SR * WIN_SEC      # 300
WIN_TGT = TARGET_SR * WIN_SEC   # 120

DIMENSION = 6
SEQ_LEN = WIN_TGT

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

# HARSense raw activity -> unified benchmark activity id
HARSENSE_TO_BENCH = {
    "sitting": 0,
    "standing": 1,
    "upstairs": 3,
    "upstaires": 3,
    "downstairs": 4,
    "downstaires": 4,
    "walking": 5,
    "running": 6
}

GLOBAL_SUBJECT_ID_OFFSET = 8
UNKNOWN_BODY_PART_ID = 100
DATASET_ID = 1

SENSOR_COLUMNS = [
    "Acc-X", "Acc-Y", "Acc-Z",
    "RR-X", "RR-Y", "RR-Z"
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_user_id(filename):
    """
    Sub1_ Adi.csv -> 0
    ...
    Sub12_Tanzim.csv -> 11
    """
    base = os.path.splitext(filename)[0]
    m = re.search(r'Sub\s*(\d+)|Sub(\d+)', base, re.IGNORECASE)
    if m is None:
        raise ValueError(f'Cannot parse subject id from filename: {filename}')

    sid = m.group(1) if m.group(1) is not None else m.group(2)
    return int(sid) - 1


def normalize_activity_name(x):
    s = str(x).strip().lower()
    return s


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


def iter_continuous_label_segments(df, label_col="activity"):
    labels = df[label_col].to_numpy()
    if len(labels) == 0:
        return

    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield labels[start], df.iloc[start:i]
            start = i
    yield labels[start], df.iloc[start:]


def iter_full_windows(arr, win_len):
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[:n * win_len]
    for i in range(n):
        yield trunc[i * win_len:(i + 1) * win_len]


def process_harsense(dataset_path, out_root, version):
    data_list = []
    label_list = []

    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith('.csv')])

    for fname in files:
        file_path = os.path.join(dataset_path, fname)
        try:
            local_user_id = parse_user_id(fname)
        except ValueError:
            print(f'[Warning] Skip file without subject id: {fname}')
            continue
        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + local_user_id

        df = pd.read_csv(file_path)

        # normalize column names by stripping spaces
        df.columns = [str(c).strip() for c in df.columns]
        # One released HARSense file uses a typo in the accelerometer Y column.
        if "Axx-Y" in df.columns and "Acc-Y" not in df.columns:
            df = df.rename(columns={"Axx-Y": "Acc-Y"})

        if "activity" not in df.columns:
            raise ValueError(f'Missing activity column in {fname}')

        missing_cols = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing required columns in {fname}: {missing_cols}')

        df["activity"] = df["activity"].apply(normalize_activity_name)
        df = df[df["activity"].isin(HARSENSE_TO_BENCH.keys())].reset_index(drop=True)
        if len(df) == 0:
            continue

        for raw_activity, seg_df in iter_continuous_label_segments(df, label_col="activity"):
            if raw_activity not in HARSENSE_TO_BENCH:
                continue

            if len(seg_df) < WIN_SRC:
                continue

            bench_activity = HARSENSE_TO_BENCH[raw_activity]

            sensor = seg_df[SENSOR_COLUMNS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
            sensor = fill_nan_linear(sensor)

            for win in iter_full_windows(sensor, WIN_SRC):   # [300, 6]
                samp = resample_linear(win, WIN_TGT)         # [120, 6]
                data_list.append(samp.astype(np.float32))
                label_list.append([[bench_activity, global_subject_id, UNKNOWN_BODY_PART_ID, DATASET_ID]])

    if not data_list:
        raise RuntimeError('No valid samples produced. Check HARSense path / file format / labels.')

    data = np.stack(data_list, axis=0).astype(np.float32)   # [N,120,6]
    label = np.array(label_list, dtype=np.int64)            # [N,1,4]

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
    dataset_name = f'HARSENSE_{version}'

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
    data, label = process_harsense(
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
