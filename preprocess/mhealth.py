#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np


DATASET_PATH = r'MHEALTH'
SAVE_DIR = os.path.join('dataset_8', 'MHEALTH')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

SRC_SR = 50
TARGET_SR = 20
WIN_SEC = 6
WIN_SRC = SRC_SR * WIN_SEC      # 300
WIN_TGT = TARGET_SR * WIN_SEC   # 120

DIMENSION = 6
SEQ_LEN = WIN_TGT

# unified benchmark activity labels
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

# MHEALTH raw activity id -> unified benchmark activity id
# dropped classes are omitted
MHEALTH_TO_BENCH = {
    1: 1,  # Standing still -> standing
    2: 0,  # Sitting and relaxing -> sitting
    3: 2,  # Lying down -> lying
    4: 5,  # Walking -> walking
    5: 3,  # Climbing stairs -> upstairs
    10: 6, # Jogging -> running
    11: 6, # Running -> running
    12: 7, # Jump front & back -> jumping
}

# keep positions that have both acc and gyro
POSITION_ORDER = [
    "left_ankle",
    "right_lower_arm"
]
BODY_PART_IDS = [5, 1]
GLOBAL_SUBJECT_ID_OFFSET = 119
DATASET_ID = 4

# column indices in file (0-based)
# left ankle: acc(6,7,8), gyro(9,10,11)
LEFT_ANKLE_COLS = [5, 6, 7, 8, 9, 10]

# right lower arm: acc(15,16,17), gyro(18,19,20)
RIGHT_ARM_COLS = [14, 15, 16, 17, 18, 19]

POSITION_COLS = [
    LEFT_ANKLE_COLS,
    RIGHT_ARM_COLS
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_user_id(filename):
    """
    mHealth_subject1.log -> 0
    ...
    mHealth_subject10.log -> 9
    """
    base = os.path.splitext(filename)[0]
    digits = ''.join([c for c in base if c.isdigit()])
    if digits == '':
        raise ValueError(f'Cannot parse subject id from filename: {filename}')
    sid = int(digits)
    return sid - 1


def load_log_file(path):
    """
    Load whitespace-separated .log file.
    Expected shape: [T, 24]
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 24:
        raise ValueError(f'Unexpected shape {arr.shape} in {path}')
    if arr.shape[1] > 24:
        arr = arr[:, :24]
    return arr.astype(np.float32)


def fill_nan_linear(x):
    """
    Linear interpolation for NaN values column-wise.
    """
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
    """
    Resample [L_src, D] -> [tgt_len, D]
    """
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def iter_continuous_label_segments(arr, label_col=23):
    """
    Split full sequence into contiguous segments with same raw label.
    arr: [T, 24]
    yield (raw_label, segment_array)
    """
    labels = arr[:, label_col].astype(np.int32)
    if len(labels) == 0:
        return

    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            yield int(labels[start]), arr[start:i]
            start = i
    yield int(labels[start]), arr[start:]


def iter_full_windows(arr, win_len):
    """
    Non-overlapping full windows.
    """
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[:n * win_len]
    for i in range(n):
        yield trunc[i * win_len:(i + 1) * win_len]


def process_mhealth(dataset_path, out_root, version):
    data_list = []
    label_list = []

    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.log')])

    for fname in files:
        file_path = os.path.join(dataset_path, fname)
        user_id = parse_user_id(fname)
        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

        arr = load_log_file(file_path)
        if np.isnan(arr).any():
            arr[:, :-1] = fill_nan_linear(arr[:, :-1])

        # split by contiguous activity segments
        for raw_label, seg in iter_continuous_label_segments(arr, label_col=23):
            # drop null class and unsupported classes
            if raw_label not in MHEALTH_TO_BENCH:
                continue

            if seg.shape[0] < WIN_SRC:
                continue

            bench_activity = MHEALTH_TO_BENCH[raw_label]

            for win in iter_full_windows(seg, WIN_SRC):   # [300, 24]
                # for each valid position, generate one sample
                for pos_id, cols in enumerate(POSITION_COLS):
                    samp_src = win[:, cols]               # [300, 6]
                    samp_src = fill_nan_linear(samp_src)
                    samp_tgt = resample_linear(samp_src, WIN_TGT)   # [120, 6]

                    data_list.append(samp_tgt.astype(np.float32))
                    body_part_id = BODY_PART_IDS[pos_id]
                    label_list.append([[bench_activity, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError('No samples produced. Please check MHEALTH path / file format / labels.')

    data = np.stack(data_list, axis=0).astype(np.float32)   # [N, 120, 6]
    label = np.array(label_list, dtype=np.int64)            # [N, 1, 4]

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
    dataset_name = f'MHEALTH_{version}'

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
    data, label = process_mhealth(
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
