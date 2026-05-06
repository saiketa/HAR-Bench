#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

from ut_complex import (
    BODY_PART_FILE_MAP,
    DATASET_ID,
    DATASET_PATH,
    DIMENSION,
    GLOBAL_SUBJECT_ID,
    SEQ_LEN,
    TARGET_SR,
    VERSION,
    WIN_SRC,
    WIN_TGT,
    ensure_dir,
    extract_sensor_and_label,
    fill_nan_linear,
    iter_continuous_label_segments,
    iter_full_windows,
    load_ut_complex_csv,
    resample_linear,
)


SAVE_DIR = os.path.join('dataset_other', 'UT-Complex')
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

# UT-Complex other raw label -> unified benchmark label
UT_COMPLEX_OTHER_TO_BENCH = {
    11115: 19,  # bike
    11118: 32,  # type
    11119: 33,  # write
    11120: 34,  # coffee
    11121: 35,  # talk
    11122: 36,  # smoke
    11123: 37,  # eat
}


def process_ut_complex_other(dataset_path, out_root, version):
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

        keep_mask = np.isin(label_arr, list(UT_COMPLEX_OTHER_TO_BENCH.keys()))
        sensor = sensor[keep_mask]
        label_arr = label_arr[keep_mask]

        if len(label_arr) == 0:
            continue

        tmp_df = pd.DataFrame(sensor, columns=["ax", "ay", "az", "gx", "gy", "gz"])
        tmp_df["label"] = label_arr

        for raw_activity, seg_df in iter_continuous_label_segments(tmp_df, "label"):
            if raw_activity not in UT_COMPLEX_OTHER_TO_BENCH:
                continue

            if len(seg_df) < WIN_SRC:
                continue

            bench_activity = UT_COMPLEX_OTHER_TO_BENCH[raw_activity]
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
    dataset_name = f'UT-COMPLEX_OTHER_{version}'

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
    data, label = process_ut_complex_other(
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
