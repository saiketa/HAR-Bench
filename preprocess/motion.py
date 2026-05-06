#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np


DATASET_PATH = r'Motion'
SAVE_DIR = os.path.join('dataset_8', 'Motion')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')

ACTIVITY_NAMES = ["dws", "ups", "sit", "std", "wlk", "jog"]
SAMPLE_WINDOW = 20
GLOBAL_SUBJECT_ID_OFFSET = 129
BODY_PART_ID = 7
DATASET_ID = 5

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

# MotionSense raw activity -> unified benchmark activity id
MOTION_TO_BENCH = {
    "dws": 4,   # downstairs
    "ups": 3,   # upstairs
    "sit": 0,   # sitting
    "std": 1,   # standing
    "wlk": 5,   # walking
    "jog": 6    # running
}


def label_activity(name):
    for act in ACTIVITY_NAMES:
        if name.startswith(act):
            return act
    return None


def label_user(name):
    temp = name.split(".")[0]
    user_id = int(temp.split("_")[1])
    return user_id - 1


def down_sample(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []

    if float(window_sample).is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice_data = data[i: i + window, :]
            if slice_data.shape[0] == window:
                result.append(np.mean(slice_data, axis=0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 < data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice_data = data[i: i + window + 1, :]
                result.append(np.mean(slice_data, axis=0))
                i += window + 1
            else:
                slice_data = data[i: i + window, :]
                result.append(np.mean(slice_data, axis=0))
                i += window

    return np.array(result)


def load_sensor_data(path, seq_len, target_window):
    data = []
    label = []

    # 为保证 acc / gyro 顺序一致，统一排序
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for dir_name in dirs:
            raw_act = label_activity(dir_name)
            if raw_act is None:
                continue
            if raw_act not in MOTION_TO_BENCH:
                continue

            bench_activity = MOTION_TO_BENCH[raw_act]
            path_act = os.path.join(root, dir_name)

            for root_exp, dirs_exp, files_exp in os.walk(path_act):
                files_exp = sorted(files_exp)
                for name in files_exp:
                    path_exp = os.path.join(root_exp, name)
                    user_id = label_user(name)

                    sensor = np.loadtxt(path_exp, skiprows=1, delimiter=',')
                    # sensor[:, 1:] 去掉时间戳/索引列，仅保留 xyz 三轴
                    sensor_down = down_sample(sensor[:, 1:], target_window)

                    if sensor_down.shape[0] < seq_len:
                        continue

                    sensor_down = sensor_down[: sensor_down.shape[0] // seq_len * seq_len, :]
                    if sensor_down.shape[0] == 0:
                        continue

                    sensor_down = sensor_down.reshape(
                        sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1]
                    )

                    num_seg = sensor_down.shape[0]

                    # [activity, global subject, body part, dataset]
                    sensor_label = np.zeros((num_seg, 1, 4), dtype=np.int64)
                    sensor_label[:, 0, 0] = bench_activity
                    sensor_label[:, 0, 1] = GLOBAL_SUBJECT_ID_OFFSET + user_id
                    sensor_label[:, 0, 2] = BODY_PART_ID
                    sensor_label[:, 0, 3] = DATASET_ID

                    data.append(sensor_down)
                    label.append(sensor_label)

    return data, label


def preprocess(path, path_save, version, target_window=50, seq_len=120):
    data_acc, label_acc = load_sensor_data(os.path.join(path, 'B_Accelerometer_data'), seq_len, target_window)
    data_gyro, label_gyro = load_sensor_data(os.path.join(path, 'C_Gyroscope_data'), seq_len, target_window)

    if len(data_acc) == 0 or len(data_gyro) == 0:
        raise ValueError('No valid accelerometer or gyroscope data found.')

    if len(data_acc) != len(data_gyro):
        print(f'[Warning] data_acc count ({len(data_acc)}) != data_gyro count ({len(data_gyro)})')

    pair_num = min(len(data_acc), len(data_gyro))
    data = []
    label = []

    for i in range(pair_num):
        len_min = min(data_acc[i].shape[0], data_gyro[i].shape[0])

        # acc: 3 dims, gyro: 3 dims -> total 6 dims
        merged = np.concatenate(
            [data_acc[i][:len_min] * 9.8, data_gyro[i][:len_min]],
            axis=2
        )

        data.append(merged)
        label.append(label_acc[i][:len_min, :, :])

    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data.npy'), np.array(data))
    np.save(os.path.join(path_save, 'label.npy'), np.array(label))

    print('All data processed. Size: %d' % data.shape[0])
    print('Data shape:', data.shape)
    print('Label shape:', label.shape)
    print('Activity classes contained:', np.unique(label[:, 0, 0]).astype(int).tolist())
    print('Global subject ids contained:', np.unique(label[:, 0, 1]).astype(int).tolist())
    print('Body part ids contained:', np.unique(label[:, 0, 2]).astype(int).tolist())
    print('Dataset ids contained:', np.unique(label[:, 0, 3]).astype(int).tolist())

    return data, label


def update_data_config(config_path, version, data, label):
    dataset_name = f'MOTION_{version}'

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
    data, label = preprocess(
        DATASET_PATH,
        SAVE_DIR,
        VERSION,
        target_window=50,
        seq_len=120
    )

    update_data_config(
        CONFIG_PATH,
        VERSION,
        data,
        label
    )
