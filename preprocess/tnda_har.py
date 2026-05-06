#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import numpy as np


# ========= 基础配置 =========
DATASET_PATH = r'TNDA-HAR'
SAVE_DIR = os.path.join('dataset_8', 'TNDA-HAR')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

SRC_SR = 50
TARGET_SR = 20
WIN_SEC = 6
WIN_SRC = SRC_SR * WIN_SEC      # 300
WIN_TGT = TARGET_SR * WIN_SEC   # 120

GROUPS = 5
AXES_PER_GROUP = 9
KEEP_IN_GROUP = [0, 1, 2, 3, 4, 5]   # acc(3) + gyro(3)

DIMENSION = 6
SEQ_LEN = WIN_TGT
USER_SIZE = 50

# ========= 统一 benchmark 活动空间 =========
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

# 原始 TNDA-HAR 活动ID -> 名称
RAW_ACTIVITY_ID_TO_NAME = {
    1: "sitting",
    2: "standing",
    3: "lying",
    4: "upstairs",
    5: "downstairs",
    7: "walking",
    8: "running",
}

# 统一 benchmark 名称 -> 统一标签ID
BENCH_ACTIVITY_TO_ID = {name: i for i, name in enumerate(BENCH_ACTIVITY_LABEL)}

# TNDA-HAR 保留的原始 activity ids
TARGET_IDS = [1, 2, 3, 4, 5, 7, 8]

# 45 sensor columns are grouped by prefix order: arm, leg, wri, ank, bac.
POSITION_ORDER = ["right_arm", "left_knee", "right_wrist", "left_ankle", "back"]
BODY_PART_IDS = [11, 2, 1, 5, 12]
GLOBAL_SUBJECT_ID_OFFSET = 187
DATASET_ID = 9


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def subject_id_from_name(name):
    """
    Subject01 -> 0, ..., Subject50 -> 49
    """
    name = name.strip()
    if not name.lower().startswith("subject"):
        raise ValueError(f"Unexpected subject filename: {name}")

    num = "".join([c for c in name if c.isdigit()])
    if not num:
        raise ValueError(f"Cannot parse subject id from: {name}")

    sid = int(num)
    if not (1 <= sid <= 50):
        raise ValueError(f"Subject id out of range: {sid}")

    return sid - 1


def robust_read_csv_46(path):
    """
    读取 46 列 CSV，返回 ndarray [T, 46] (float32)
    前45列为传感器，最后1列为活动ID
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if not line or all((x.strip() == "" for x in line)):
                continue

            try:
                vals = [float(x.strip()) for x in line]
            except ValueError:
                if i == 0:
                    # 可能是表头
                    continue
                try:
                    vals = [float(x.strip()) for x in line[0].split(";")]
                except Exception as e:
                    raise ValueError(f"Bad numeric row in {path}, line {i+1}: {line}") from e

            rows.append(vals)

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 46:
        raise ValueError(f"Unexpected shape {arr.shape} in {path}")

    if arr.shape[1] > 46:
        arr = arr[:, :46]

    return arr


def fill_nan_linear(x):
    """
    对每一列线性插值补 NaN
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


def iter_full_windows(arr, win_len):
    """
    无重叠整窗切分
    """
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return

    trunc = arr[:n * win_len, :]
    for i in range(n):
        yield trunc[i * win_len:(i + 1) * win_len, :]


def resample_linear(win_src, tgt_len):
    """
    线性重采样：L_src -> tgt_len
    """
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def split_45_to_5x6(win_120x45):
    """
    45列 -> 5组 × 9轴，每组仅保留前6轴(acc+gyro)
    返回 5 个 (120, 6)
    """
    out = []
    for g in range(GROUPS):
        start = g * AXES_PER_GROUP
        cols9 = win_120x45[:, start:start + AXES_PER_GROUP]
        cols6 = cols9[:, KEEP_IN_GROUP]
        out.append(cols6.astype(np.float32))
    return out


def process_tnda_har(tnda_root, out_root, version):
    data_list = []
    label_list = []

    for fname in sorted(os.listdir(tnda_root)):
        if not fname.lower().endswith(".csv"):
            continue

        subj_path = os.path.join(tnda_root, fname)
        user_id = subject_id_from_name(os.path.splitext(fname)[0])
        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

        arr = robust_read_csv_46(subj_path)   # [T, 46]

        if np.isnan(arr).any():
            arr[:, :45] = fill_nan_linear(arr[:, :45])

        act_col = arr[:, 45].astype(np.int32)

        keep_mask = np.isin(act_col, TARGET_IDS)
        if not keep_mask.any():
            continue

        arr = arr[keep_mask, :]
        act_col = act_col[keep_mask]

        for raw_id in TARGET_IDS:
            seg = arr[act_col == raw_id, :]   # [T, 46]
            if seg.shape[0] < WIN_SRC:
                continue

            data_45 = seg[:, :45]

            # 6秒无重叠窗：300点@50Hz
            for win in iter_full_windows(data_45, WIN_SRC):
                win = fill_nan_linear(win)
                win120 = resample_linear(win, WIN_TGT)   # [120, 45]

                samples = split_45_to_5x6(win120)        # 5 * [120, 6]
                for pos_id, samp in enumerate(samples):
                    raw_act_name = RAW_ACTIVITY_ID_TO_NAME[raw_id]
                    bench_act = BENCH_ACTIVITY_TO_ID[raw_act_name]
                    body_part_id = BODY_PART_IDS[pos_id]

                    data_list.append(samp)
                    label_list.append([[bench_act, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check TNDA-HAR path / CSV format / filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)     # [N, 120, 6]
    label = np.array(label_list, dtype=np.int64)              # [N, 1, 4]

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
    dataset_name = f'TNDA-HAR_{version}'

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
    data, label = process_tnda_har(
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
