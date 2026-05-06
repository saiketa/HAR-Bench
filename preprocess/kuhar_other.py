import os
import json
import numpy as np

from kuhar import (
    BODY_PART_ID,
    DATA_FILENAME,
    DATASET_DIRNAME,
    DATASET_ID,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    LABEL_FILENAME,
    SEQ_LEN,
    TARGET_SR,
    WIN_SAMPLES_SRC,
    WIN_SAMPLES_TGT,
    collect_all_users,
    drop_timestamps_and_convert_units,
    ensure_dir,
    fill_nan_linear,
    iter_full_windows,
    parse_base_userid,
    resample_linear,
    robust_read_csv_8,
)


FOLDER_TO_ACTIVITY_ID = {
    "2.Talk-sit": 9,
    "3.Talk-stand": 10,
    "4.Stand-sit": 11,
    "6.Lay-stand": 12,
    "7.Pick": 13,
    "9.Push-up": 208,
    "10.Sit-up": 209,
    "12.Walk-backwards": 14,
    "13.Walk-circle": 15,
    "17.Table-tennis": 210,
}
TARGET_FOLDERS = set(FOLDER_TO_ACTIVITY_ID.keys())
CONFIG_KEY = f"KU-HAR_OTHER_{TARGET_SR}_{SEQ_LEN}"


def process_kuhar_other(kuhar_root, out_root):
    user_map = collect_all_users(kuhar_root)
    print(f"[INFO] Collected {len(user_map)} unique users.")

    data_list = []
    label_list = []

    for folder in sorted(os.listdir(kuhar_root)):
        if folder not in TARGET_FOLDERS:
            continue

        act_id = FOLDER_TO_ACTIVITY_ID[folder]
        fdir = os.path.join(kuhar_root, folder)
        if not os.path.isdir(fdir):
            continue

        for fname in sorted(os.listdir(fdir)):
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(fdir, fname)
            arr = robust_read_csv_8(fpath)

            base_uid = parse_base_userid(os.path.splitext(fname)[0])
            if base_uid not in user_map:
                user_map[base_uid] = len(user_map)
            user_id = user_map[base_uid]
            global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

            if arr.shape[0] < WIN_SAMPLES_SRC:
                continue

            for win in iter_full_windows(arr, WIN_SAMPLES_SRC):
                win = fill_nan_linear(win)
                win120 = resample_linear(win, WIN_SAMPLES_TGT)
                samp = drop_timestamps_and_convert_units(win120)

                data_list.append(samp)
                label_list.append([[act_id, global_subject_id, BODY_PART_ID, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check KU-HAR path/folder names/CSV format.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)

    print(f"[INFO] Final: data={data.shape}, labels={labels.shape}, users={len(user_map)}")

    out_dir = os.path.join(out_root, DATASET_DIRNAME)
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)

    return data.shape[0], out_dir, len(user_map)


def update_data_config(config_path, total_size, user_size):
    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            print("[WARN] data_config.json invalid. Recreating.")

    cfg[CONFIG_KEY] = {
        "sr": TARGET_SR,
        "seq_len": SEQ_LEN,
        "dimension": DIMENSION,
        "activity_label_index": 0,
        "global_subject_label_index": 1,
        "body_part_label_index": 2,
        "dataset_label_index": 3,
        "size": int(total_size),
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Updated data_config: {CONFIG_KEY} (size={total_size}, users={user_size})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare KU-HAR other activities to npy (20Hz, 6x120)."
    )
    parser.add_argument(
        "--kuhar_root",
        type=str,
        default="./KU-HAR/1.Raw_time_domian_data",
        help="Path to KU-HAR root containing activity folders",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset_other",
        help="Output root; will create dataset_other/KU-HAR",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_other", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir, user_size = process_kuhar_other(args.kuhar_root, args.dataset_root)
    update_data_config(args.config_path, total, user_size)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config path: {args.config_path}")
