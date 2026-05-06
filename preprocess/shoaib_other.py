# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd

from shoaib import (
    ACT_LABELS,
    BODY_PART_IDS,
    DATASET_ID,
    GLOBAL_SUBJECT_ID_OFFSET,
    down_sample,
    label_name_to_index,
    participant_sort_key,
)


DATASET_PATH = r'Shoaib'
SAVE_DIR = os.path.join('dataset_other', 'Shoaib')
VERSION = r'20_120'
CONFIG_PATH = os.path.join('dataset_other', 'data_config.json')

# Shoaib other activity -> unified benchmark activity id
SHOAIB_OTHER_TO_BENCH = {
    "biking": 19
}


def preprocess(path, path_save, version, target_window=50, seq_len=120, position_num=5):
    data_all = []
    label_all = []

    user_id = 0

    for root, dirs, files in os.walk(path):
        files = sorted(files, key=participant_sort_key)
        for file_name in files:
            if 'Participant' not in file_name:
                continue

            file_path = os.path.join(root, file_name)
            exp = pd.read_csv(file_path, skiprows=1)

            labels_activity_name = exp.iloc[:, -1].astype(str).str.strip().to_numpy()
            labels_activity_idx = label_name_to_index(labels_activity_name)

            global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

            print(f'Processing user file: {file_name} -> assigned global_subject_id={global_subject_id}')

            for raw_act_name, bench_activity in SHOAIB_OTHER_TO_BENCH.items():
                raw_act_idx = ACT_LABELS.index(raw_act_name)
                exp_act = exp.iloc[labels_activity_idx == raw_act_idx, :]

                for pos_id in range(position_num):
                    index_6d = np.array([1, 2, 3, 10, 11, 12]) + pos_id * 14

                    if np.max(index_6d) >= exp_act.shape[1]:
                        print(f'[Warning] position index out of range in {file_name}, position={pos_id}')
                        continue

                    exp_pos = exp_act.iloc[:, index_6d].to_numpy(dtype=np.float32)

                    print(
                        "User-%s, activity-%s, position-%d: num-%d"
                        % (file_name, raw_act_name, pos_id, exp_pos.shape[0])
                    )

                    if exp_pos.shape[0] == 0:
                        continue

                    exp_pos_down = down_sample(exp_pos, target_window)

                    if exp_pos_down.shape[0] < seq_len:
                        continue

                    sensor_down = exp_pos_down[: exp_pos_down.shape[0] // seq_len * seq_len, :]
                    if sensor_down.shape[0] == 0:
                        continue

                    sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])

                    num_seg = sensor_down.shape[0]

                    # [activity, global subject, body part, dataset]
                    sensor_label = np.zeros((num_seg, 1, 4), dtype=np.int64)
                    sensor_label[:, 0, 0] = bench_activity
                    sensor_label[:, 0, 1] = global_subject_id
                    sensor_label[:, 0, 2] = BODY_PART_IDS[pos_id]
                    sensor_label[:, 0, 3] = DATASET_ID

                    data_all.append(sensor_down)
                    label_all.append(sensor_label)

            user_id += 1

    if len(data_all) == 0:
        raise ValueError('No valid data generated. Please check dataset path and file format.')

    data_all = np.concatenate(data_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)

    os.makedirs(path_save, exist_ok=True)
    np.save(os.path.join(path_save, 'data.npy'), np.array(data_all))
    np.save(os.path.join(path_save, 'label.npy'), np.array(label_all))

    print('All data processed. Size: %d' % data_all.shape[0])
    print('Data shape:', data_all.shape)
    print('Label shape:', label_all.shape)
    print('Activity classes contained:', np.unique(label_all[:, 0, 0]).astype(int).tolist())
    print('Global subject ids contained:', np.unique(label_all[:, 0, 1]).astype(int).tolist())
    print('Body part ids contained:', np.unique(label_all[:, 0, 2]).astype(int).tolist())
    print('Dataset ids contained:', np.unique(label_all[:, 0, 3]).astype(int).tolist())

    return data_all, label_all


def update_data_config(config_path, version, data, label):
    dataset_name = f'SHOAIB_OTHER_{version}'

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
        seq_len=120,
        position_num=5
    )

    update_data_config(
        CONFIG_PATH,
        VERSION,
        data,
        label
    )
