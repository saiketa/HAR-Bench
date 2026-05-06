import os
import json
import re
import numpy as np

from dsads import (
    AXES_PER_GROUP,
    BODY_PART_IDS,
    DATASET_ID,
    DATA_FILENAME,
    DIMENSION,
    GROUPS,
    KEEP_IDXS_WITHIN_GROUP,
    LABEL_FILENAME,
    SEQ_LEN,
    TARGET_SR,
    USER_SIZE,
    WIN_SAMPLES_SRC,
    WIN_SAMPLES_TGT,
    concat_subject_activity,
    ensure_dir,
    resample_150_to_120_linear,
    split_45cols_to_5x6,
    split_to_windows_6s,
)


# =========================
# DSADS 其他活动文件夹 -> 统一活动编号
# =========================
OTHER_ACTIVITY_FOLDERS = {
    "a04": 8,    # lying on right side
    "a07": 101,  # standing still in elevator
    "a08": 102,  # move in elevator
    "a10": 201,  # walking on flat treadmill
    "a11": 202,  # walking on inclined treadmill
    "a13": 203,  # exercising on stepper
    "a14": 204,  # exercising on a cross trainer
    "a15": 205,  # cycling on exercise bike
    "a16": 205,  # cycling on exercise bike
    "a17": 206,  # rowing
    "a19": 207,  # playing basketball
}

CONFIG_KEY = f"DSADS_OTHER_{TARGET_SR}_{SEQ_LEN}"


def process_dsads_other(dsads_root, out_root):
    """
    dsads_root:
      └─ a01 ... a19
           └─ p1 ... p8
                └─ s01.txt ... s60.txt

    out_root:
      └─ DSADS
           ├─ data.npy
           └─ label.npy
    """
    data_list = []
    label_list = []

    for a_folder in sorted(OTHER_ACTIVITY_FOLDERS):
        act_id = OTHER_ACTIVITY_FOLDERS[a_folder]
        a_path = os.path.join(dsads_root, a_folder)

        if not os.path.isdir(a_path):
            print(f"[WARN] Activity folder not found: {a_path}, skip")
            continue

        subjects = sorted([d for d in os.listdir(a_path) if d.lower().startswith("p")])

        for s in subjects:
            try:
                subj_idx = int(re.sub(r"[^0-9]", "", s)) - 1
            except Exception:
                print(f"[WARN] Unexpected subject name {s} in {a_path}, skip")
                continue

            if not (0 <= subj_idx < USER_SIZE):
                print(f"[WARN] Subject id out of range: {s} in {a_path}, skip")
                continue

            s_path = os.path.join(a_path, s)

            txts = [os.path.join(s_path, f"s{idx:02d}.txt") for idx in range(1, 61)]
            missing = [t for t in txts if not os.path.isfile(t)]
            if missing:
                print(f"[WARN] Missing {len(missing)} files under {s_path}, skip this subject")
                continue

            arr = concat_subject_activity(txts)
            wins = split_to_windows_6s(arr)

            for w in range(wins.shape[0]):
                win120x45 = resample_150_to_120_linear(wins[w])
                five_samples = split_45cols_to_5x6(win120x45)

                for pos_id, samp in enumerate(five_samples):
                    data_list.append(samp)
                    body_part_id = BODY_PART_IDS[pos_id]
                    global_subject_id = subj_idx
                    label_list.append([[act_id, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples were produced. Please check DSADS path and structure.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)

    print(f"[INFO] Final dataset shape: data={data.shape}, labels={labels.shape}")

    out_dir = os.path.join(out_root, "DSADS")
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)

    return data.shape[0], out_dir


def update_data_config(config_path, total_size):
    cfg = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                cfg = json.load(f)
            except json.JSONDecodeError:
                print("[WARN] data_config.json is not valid JSON. Recreating it.")

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

    print(f"[INFO] Updated config: '{CONFIG_KEY}' (size={total_size})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare DSADS other activities to npy format (20Hz, 6x120)."
    )
    parser.add_argument(
        "--dsads_root",
        type=str,
        default="./DSADS",
        help="Path to DSADS root containing a01..a19 folders",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset_other",
        help="Output root; will create dataset_other/DSADS",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_other", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_dsads_other(args.dsads_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved data to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config updated at: {args.config_path}")
