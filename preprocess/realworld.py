#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import zipfile
import numpy as np
import pandas as pd


DATASET_PATH = r'RealWorld'
SAVE_DIR = os.path.join('dataset_8', 'RealWorld')
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

# RealWorld raw activity -> unified benchmark activity id
REALWORLD_TO_BENCH = {
    "climbingdown": 4,
    "climbingup": 3,
    "jumping": 7,
    "lying": 2,
    "standing": 1,
    "sitting": 0,
    "running": 6,
    "jogging": 6,
    "walking": 5
}

POSITION_ORDER = [
    "chest",
    "forearm",
    "head",
    "shin",
    "thigh",
    "upperarm",
    "waist"
]
POSITION_TO_ID = {
    "chest": 0,
    "forearm": 1,
    "head": 9,
    "shin": 4,
    "thigh": 2,
    "upperarm": 11,
    "waist": 8,
}
GLOBAL_SUBJECT_ID_OFFSET = 162
DATASET_ID = 7


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_user_id(folder_name):
    """
    proband1 -> 0
    ...
    proband15 -> 14
    """
    m = re.search(r'proband(\d+)', folder_name.lower())
    if m is None:
        raise ValueError(f'Cannot parse user id from folder name: {folder_name}')
    return int(m.group(1)) - 1


def normalize_activity_name(name):
    """
    Normalize activity names from zip / csv names.
    """
    name = name.lower().replace(" ", "").replace("_", "")
    if "climbingdown" in name:
        return "climbingdown"
    if "climbingup" in name:
        return "climbingup"
    if "jumping" in name:
        return "jumping"
    if "lying" in name:
        return "lying"
    if "standing" in name:
        return "standing"
    if "sitting" in name:
        return "sitting"
    if "running" in name:
        return "running"
    if "jogging" in name:
        return "jogging"
    if "walking" in name:
        return "walking"
    return None


def normalize_position_name(name):
    """
    Normalize position names from csv filenames.
    """
    s = name.lower().replace(" ", "").replace("_", "")
    if "chest" in s:
        return "chest"
    if "forearm" in s:
        return "forearm"
    if "head" in s:
        return "head"
    if "shin" in s:
        return "shin"
    if "thigh" in s:
        return "thigh"
    if "upperarm" in s:
        return "upperarm"
    if "waist" in s:
        return "waist"
    return None


def load_csv_from_zip(zip_path, member_name):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(member_name) as f:
            raw = f.read()

    # try pandas robust parse
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=';')
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), header=None)

    return df


def extract_xyz(df):
    """
    Extract xyz columns robustly from RealWorld csv.
    Many variants exist:
    - columns named x,y,z
    - columns named attr_x, attr_y, attr_z
    - timestamp + x,y,z
    - headerless 3/4-column file
    """
    cols_lower = [str(c).strip().lower() for c in df.columns]

    # named columns
    xyz_candidates = [
        ("x", "y", "z"),
        ("attr_x", "attr_y", "attr_z"),
        ("accelerometer_x", "accelerometer_y", "accelerometer_z"),
        ("gyroscope_x", "gyroscope_y", "gyroscope_z"),
    ]
    for cx, cy, cz in xyz_candidates:
        if cx in cols_lower and cy in cols_lower and cz in cols_lower:
            ix = cols_lower.index(cx)
            iy = cols_lower.index(cy)
            iz = cols_lower.index(cz)
            arr = df.iloc[:, [ix, iy, iz]].to_numpy(dtype=np.float32)
            return arr

    # fallback: choose last 3 numeric columns
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    valid_cols = [i for i in range(numeric_df.shape[1]) if numeric_df.iloc[:, i].notna().sum() > 0]
    if len(valid_cols) >= 3:
        arr = numeric_df.iloc[:, valid_cols[-3:]].to_numpy(dtype=np.float32)
        return arr

    raise ValueError(f'Cannot extract xyz columns from dataframe with columns: {list(df.columns)}')


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


def iter_full_windows(arr, win_len):
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[:n * win_len]
    for i in range(n):
        yield trunc[i * win_len:(i + 1) * win_len]


def list_csv_members(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return [n for n in zf.namelist() if n.lower().endswith('.csv')]


def group_members_by_position(members):
    """
    Return dict[position] = member_name
    """
    out = {}
    for m in members:
        pos = normalize_position_name(os.path.basename(m))
        if pos is not None:
            out[pos] = m
    return out


def collect_zip_pairs(data_dir):
    """
    Scan data dir and return list of (activity_name, acc_zip, gyr_zip)
    """
    files = sorted(os.listdir(data_dir))
    acc_map = {}
    gyr_map = {}

    for fname in files:
        low = fname.lower()
        if not low.endswith('.zip'):
            continue
        if '_csv.zip' not in low:
            continue

        act = normalize_activity_name(low)
        if act is None:
            continue

        full_path = os.path.join(data_dir, fname)
        if low.startswith('acc_'):
            acc_map[act] = full_path
        elif low.startswith('gyr_') or low.startswith('gyro_'):
            gyr_map[act] = full_path

    common_acts = sorted(set(acc_map.keys()) & set(gyr_map.keys()))
    pairs = [(act, acc_map[act], gyr_map[act]) for act in common_acts]
    return pairs


def process_realworld(dataset_path, out_root, version):
    data_list = []
    label_list = []

    proband_dirs = sorted([d for d in os.listdir(dataset_path) if d.lower().startswith('proband')])

    for proband in proband_dirs:
        user_id = parse_user_id(proband)
        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id
        data_dir = os.path.join(dataset_path, proband, 'data')
        if not os.path.isdir(data_dir):
            continue

        zip_pairs = collect_zip_pairs(data_dir)

        for raw_act, acc_zip, gyr_zip in zip_pairs:
            if raw_act not in REALWORLD_TO_BENCH:
                continue

            bench_activity = REALWORLD_TO_BENCH[raw_act]

            acc_members = group_members_by_position(list_csv_members(acc_zip))
            gyr_members = group_members_by_position(list_csv_members(gyr_zip))

            common_positions = sorted(set(acc_members.keys()) & set(gyr_members.keys()))

            for pos_name in common_positions:
                if pos_name not in POSITION_TO_ID:
                    continue

                body_part_id = POSITION_TO_ID[pos_name]

                try:
                    acc_df = load_csv_from_zip(acc_zip, acc_members[pos_name])
                    gyr_df = load_csv_from_zip(gyr_zip, gyr_members[pos_name])

                    acc_xyz = extract_xyz(acc_df)
                    gyr_xyz = extract_xyz(gyr_df)
                except Exception as e:
                    print(f'[Warning] Skip {proband} {raw_act} {pos_name}: {e}')
                    continue

                acc_xyz = fill_nan_linear(acc_xyz.astype(np.float32))
                gyr_xyz = fill_nan_linear(gyr_xyz.astype(np.float32))

                length = min(len(acc_xyz), len(gyr_xyz))
                if length < WIN_SRC:
                    continue

                merged = np.concatenate([acc_xyz[:length], gyr_xyz[:length]], axis=1)   # [T,6]

                for win in iter_full_windows(merged, WIN_SRC):   # [300,6]
                    samp = resample_linear(win, WIN_TGT)         # [120,6]
                    data_list.append(samp.astype(np.float32))
                    label_list.append([[bench_activity, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError('No valid samples produced. Check RealWorld path / zip structure / csv format.')

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
    dataset_name = f'REALWORLD_{version}'

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
    data, label = process_realworld(
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
