#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np


DATASET_PATH = r'UCI/RawData'

# 输出目录
SAVE_DIR = os.path.join('dataset_8', 'UCI')
VERSION = r'20_120'

# 根目录下的 data_config.json
CONFIG_PATH = os.path.join('dataset_8', 'data_config.json')
GLOBAL_SUBJECT_ID_OFFSET = 237
BODY_PART_ID = 8
DATASET_ID = 10

# UCI raw label -> unified benchmark label
# benchmark:
# 0 sitting
# 1 standing
# 2 lying
# 3 upstairs
# 4 downstairs
# 5 walking
# 6 running
# 7 jumping
UCI_TO_BENCH_ACTIVITY = {
    1: 5,  # WALKING -> walking
    2: 3,  # WALKING_UPSTAIRS -> upstairs
    3: 4,  # WALKING_DOWNSTAIRS -> downstairs
    4: 0,  # SITTING -> sitting
    5: 1,  # STANDING -> standing
    6: 2,  # LAYING -> lying
}

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


def down_sample(data, window_sample, start, end):
    result = []
    if float(window_sample).is_integer():
        window = int(window_sample)
        for i in range(int(start), int(end) - window, window):
            slice_data = data[i: i + window, :]
            result.append(np.mean(slice_data, axis=0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = int(start)
        while int(start) <= i + window + 1 < int(end):
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


def preprocess(path, path_save, version, raw_sr=50, target_sr=20, seq_len=120):
    labels = np.loadtxt(os.path.join(path, 'labels.txt'), delimiter=' ')
    data_all = []
    label_all = []
    window_sample = raw_sr / target_sr

    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.startswith('acc'):
                continue

            tags = name.split('.')[0].split('_')
            if len(tags) < 3:
                print(f'[Warning] Skip invalid filename: {name}')
                continue

            exp_num = int(tags[1][-2:])
            exp_user = int(tags[2][-2:])

            acc_file = os.path.join(root, name)
            gyro_file = os.path.join(root, 'gyro' + name[3:])

            if not os.path.exists(gyro_file):
                print(f'[Warning] Missing gyro file for {acc_file}')
                continue

            exp_data_acc = np.loadtxt(acc_file, delimiter=' ') * 9.80665
            exp_data_gyro = np.loadtxt(gyro_file, delimiter=' ')
            sensor_data = np.concatenate([exp_data_acc, exp_data_gyro], axis=1)

            # labels.txt: [experiment_id, user_id, activity_id, start, end]
            label_index = (labels[:, 0] == exp_num) & (labels[:, 1] == exp_user)
            label_stat = labels[label_index, :]

            for i in range(label_stat.shape[0]):
                uci_activity = int(label_stat[i, 2])

                # 只保留可对齐到统一benchmark的6类
                if uci_activity not in UCI_TO_BENCH_ACTIVITY:
                    continue

                bench_activity = UCI_TO_BENCH_ACTIVITY[uci_activity]
                index_start = int(label_stat[i, 3])
                index_end = int(label_stat[i, 4])

                seg_data = down_sample(sensor_data, window_sample, index_start, index_end)

                if seg_data.shape[0] < seq_len:
                    continue

                seg_data = seg_data[: seg_data.shape[0] // seq_len * seq_len, :]
                seg_data = seg_data.reshape(seg_data.shape[0] // seq_len, seq_len, seg_data.shape[1])

                # label: [activity, global subject, body part, dataset]
                num_seg = seg_data.shape[0]

                seg_label = np.zeros((num_seg, 1, 4), dtype=np.int64)
                
                seg_label[:, 0, 0] = bench_activity
                seg_label[:, 0, 1] = GLOBAL_SUBJECT_ID_OFFSET + exp_user - 1
                seg_label[:, 0, 2] = BODY_PART_ID
                seg_label[:, 0, 3] = DATASET_ID

                data_all.append(seg_data)
                label_all.append(seg_label)

    if len(data_all) == 0:
        raise ValueError('No valid data segments found. Please check dataset path and file structure.')

    data_all = np.concatenate(data_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data.npy'), np.array(data_all))
    np.save(os.path.join(path_save, 'label.npy'), np.array(label_all))

    print('All data processed. Size: %d' % data_all.shape[0])
    print('Activity classes contained:', np.unique(label_all[:, 0, 0]).astype(int).tolist())
    print('Global subject ids contained:', np.unique(label_all[:, 0, 1]).astype(int).tolist())
    print('Body part ids contained:', np.unique(label_all[:, 0, 2]).astype(int).tolist())
    print('Dataset ids contained:', np.unique(label_all[:, 0, 3]).astype(int).tolist())

    return data_all, label_all


def update_data_config(config_path, version, data, label):
    dataset_name = f'UCI_{version}'

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
        path=DATASET_PATH,
        path_save=SAVE_DIR,
        version=VERSION,
        raw_sr=50,
        target_sr=20,
        seq_len=120
    )

    update_data_config(
        config_path=CONFIG_PATH,
        version=VERSION,
        data=data,
        label=label
    )
