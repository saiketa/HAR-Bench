#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd


DATASET_PATH = os.path.join('HHAR', 'Activity recognition exp')
SAVE_DIR = os.path.join('dataset_8', 'HHAR')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

# unified benchmark labels
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

# HHAR raw activity -> unified benchmark activity id
HHAR_TO_BENCH = {
    "sit": 0,
    "stand": 1,
    "stairsup": 3,
    "stairsdown": 4,
    "walk": 5
    # "bike" is dropped
}

USER_TO_GLOBAL_ID = {name: 20 + i for i, name in enumerate(["a", "b", "c", "d", "e", "f", "g", "h", "i"])}
BODY_PART_ID = 8
DATASET_ID = 2
SENSOR_PAIRS = [
    ("Phones_accelerometer.csv", "Phones_gyroscope.csv"),
    ("Watch_accelerometer.csv", "Watch_gyroscope.csv"),
]


def extract_sensor(data, time_index, time_tag, window_time):
    index = time_index
    while index < len(data) and abs(data.iloc[index]['Creation_Time'] - time_tag) < window_time:
        index += 1

    if index == time_index:
        return None, index

    data_slice = data.iloc[time_index:index]

    if data_slice['User'].nunique() > 1 or data_slice['gt'].nunique() > 1 or data_slice['Device'].nunique() > 1:
        return None, index

    data_sensor = data_slice[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    sensor = np.mean(data_sensor, axis=0)

    label = data_slice[['User', 'Device', 'gt']].iloc[0].astype(str).values
    return np.concatenate([sensor, label]), index


def process_sensor_pair(path, acc_file, gyro_file, window_time, seq_len, jump):
    acc_path = os.path.join(path, acc_file)
    gyro_path = os.path.join(path, gyro_file)

    accs = pd.read_csv(acc_path)
    gyros = pd.read_csv(gyro_path)

    time_tag = min(accs.iloc[0]['Creation_Time'], gyros.iloc[0]['Creation_Time'])
    time_index = [0, 0]   # [acc_index, gyro_index]

    window_num = 0
    data_segments = []
    data_temp = []

    time_window_us = window_time * pow(10, 6)

    while time_index[0] < len(accs) and time_index[1] < len(gyros):
        acc, time_index_new_acc = extract_sensor(accs, time_index[0], time_tag, window_time=time_window_us)
        gyro, time_index_new_gyro = extract_sensor(gyros, time_index[1], time_tag, window_time=time_window_us)
        time_index = [time_index_new_acc, time_index_new_gyro]

        if acc is not None and gyro is not None and np.all(acc[-3:] == gyro[-3:]):
            raw_user = str(acc[-3])
            raw_device = str(acc[-2])
            raw_gt = str(acc[-1])

            if raw_gt not in HHAR_TO_BENCH:
                if window_num > 0:
                    data_temp.clear()
                    window_num = 0
                time_tag += time_window_us
                continue

            if raw_user not in USER_TO_GLOBAL_ID:
                if window_num > 0:
                    data_temp.clear()
                    window_num = 0
                time_tag += time_window_us
                continue

            bench_activity = HHAR_TO_BENCH[raw_gt]
            global_subject_id = USER_TO_GLOBAL_ID[raw_user]

            time_tag += time_window_us
            window_num += 1

            # 6-dim sensor + [activity, global subject, body part, dataset]
            data_temp.append(np.array([
                float(acc[0]), float(acc[1]), float(acc[2]),
                float(gyro[0]), float(gyro[1]), float(gyro[2]),
                int(bench_activity), int(global_subject_id), int(BODY_PART_ID), int(DATASET_ID)
            ], dtype=np.float32))

            if window_num == seq_len:
                data_segments.append(np.array(data_temp, dtype=np.float32))

                if jump == 0:
                    data_temp.clear()
                    window_num = 0
                else:
                    data_temp = data_temp[-jump:]
                    window_num -= jump

        else:
            if window_num > 0:
                data_temp.clear()
                window_num = 0

            if time_index[0] < len(accs) and time_index[1] < len(gyros):
                time_tag = min(
                    accs.iloc[time_index[0]]['Creation_Time'],
                    gyros.iloc[time_index[1]]['Creation_Time']
                )
            else:
                break

    if len(data_segments) == 0:
        print(f'[Warning] No valid HHAR windows generated for {acc_file} / {gyro_file}.')
        return None, None

    data_raw = np.array(data_segments, dtype=np.float32)
    data_new = data_raw[:, :, :6].astype(np.float32)
    label_raw = data_raw[:, 0, 6:].astype(np.int64)
    label_new = label_raw[:, np.newaxis, :]

    print(f'Processed {acc_file} / {gyro_file}: {data_new.shape[0]} samples')
    return data_new, label_new


def preprocess_hhar(path, path_save, version, config_path,
                    window_time=50, seq_len=120, jump=0):
    data_parts = []
    label_parts = []

    for acc_file, gyro_file in SENSOR_PAIRS:
        data_part, label_part = process_sensor_pair(
            path=path,
            acc_file=acc_file,
            gyro_file=gyro_file,
            window_time=window_time,
            seq_len=seq_len,
            jump=jump
        )
        if data_part is not None:
            data_parts.append(data_part)
            label_parts.append(label_part)

    if not data_parts:
        raise ValueError('No valid HHAR windows generated. Please check dataset files and path.')

    data_new = np.concatenate(data_parts, axis=0)
    label_new = np.concatenate(label_parts, axis=0)

    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data.npy'), np.array(data_new))
    np.save(os.path.join(path_save, 'label.npy'), np.array(label_new))

    print('All data processed. Size: %d' % data_new.shape[0])
    print('Data shape:', data_new.shape)
    print('Label shape:', label_new.shape)
    print('Activity classes contained:', np.unique(label_new[:, 0, 0]).astype(int).tolist())
    print('Global subject ids contained:', np.unique(label_new[:, 0, 1]).astype(int).tolist())
    print('Body part ids contained:', np.unique(label_new[:, 0, 2]).astype(int).tolist())
    print('Dataset ids contained:', np.unique(label_new[:, 0, 3]).astype(int).tolist())

    update_data_config(
        config_path=config_path,
        version=version,
        data=data_new,
        label=label_new
    )

    return data_new, label_new


def update_data_config(config_path, version, data, label):
    dataset_name = f'HHAR_{version}'
    size = int(data.shape[0])

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}

    config[dataset_name] = {
        "sr": 20,
        "seq_len": 120,
        "dimension": 6,
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
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f'Updated {config_path} with key: {dataset_name}')


if __name__ == '__main__':
    data, label = preprocess_hhar(
        DATASET_PATH,
        SAVE_DIR,
        VERSION,
        CONFIG_PATH,
        window_time=50,
        seq_len=120,
        jump=0
    )
