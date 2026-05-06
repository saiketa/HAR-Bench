#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from uci import (
    BODY_PART_ID,
    DATASET_ID,
    DATASET_PATH,
    GLOBAL_SUBJECT_ID_OFFSET,
    VERSION,
    down_sample,
)


SAVE_DIR = os.path.join('dataset_other', 'UCI')
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

# UCI other raw activity id -> unified benchmark activity id
UCI_OTHER_TO_BENCH_ACTIVITY = {
    8: 27,   # SIT_TO_STAND
    9: 28,   # SIT_TO_LIE
    10: 29,  # LIE_TO_SIT
    11: 30,  # STAND_TO_LIE
    12: 31,  # LIE_TO_STAND
}


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

            label_index = (labels[:, 0] == exp_num) & (labels[:, 1] == exp_user)
            label_stat = labels[label_index, :]

            for i in range(label_stat.shape[0]):
                uci_activity = int(label_stat[i, 2])

                if uci_activity not in UCI_OTHER_TO_BENCH_ACTIVITY:
                    continue

                bench_activity = UCI_OTHER_TO_BENCH_ACTIVITY[uci_activity]
                index_start = int(label_stat[i, 3])
                index_end = int(label_stat[i, 4])

                seg_data = down_sample(sensor_data, window_sample, index_start, index_end)

                if seg_data.shape[0] < seq_len:
                    continue

                seg_data = seg_data[: seg_data.shape[0] // seq_len * seq_len, :]
                seg_data = seg_data.reshape(seg_data.shape[0] // seq_len, seq_len, seg_data.shape[1])

                num_seg = seg_data.shape[0]

                # label: [activity, global subject, body part, dataset]
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
    dataset_name = f'UCI_OTHER_{version}'

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
