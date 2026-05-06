#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from mhealth import (
    BODY_PART_IDS,
    DATASET_ID,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    POSITION_COLS,
    SEQ_LEN,
    TARGET_SR,
    WIN_SRC,
    WIN_TGT,
    ensure_dir,
    fill_nan_linear,
    iter_continuous_label_segments,
    iter_full_windows,
    load_log_file,
    parse_user_id,
    resample_linear,
)


DATASET_PATH = r'MHEALTH'
SAVE_DIR = os.path.join('dataset_other', 'MHEALTH')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

MHEALTH_OTHER_TO_ACTIVITY = {
    6: 16,  # Waist bends forward
    7: 17,  # Frontal elevation of arms
    8: 18,  # Knees bending (crouching)
    9: 19,  # Cycling
}


def process_mhealth_other(dataset_path, out_root, version):
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

        for raw_label, seg in iter_continuous_label_segments(arr, label_col=23):
            if raw_label not in MHEALTH_OTHER_TO_ACTIVITY:
                continue

            if seg.shape[0] < WIN_SRC:
                continue

            activity_id = MHEALTH_OTHER_TO_ACTIVITY[raw_label]

            for win in iter_full_windows(seg, WIN_SRC):
                for pos_id, cols in enumerate(POSITION_COLS):
                    samp_src = win[:, cols]
                    samp_src = fill_nan_linear(samp_src)
                    samp_tgt = resample_linear(samp_src, WIN_TGT)

                    data_list.append(samp_tgt.astype(np.float32))
                    body_part_id = BODY_PART_IDS[pos_id]
                    label_list.append([[activity_id, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError('No samples produced. Please check MHEALTH path / file format / labels.')

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
    dataset_name = f'MHEALTH_OTHER_{version}'

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
    data, label = process_mhealth_other(
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
