import os
import json
import numpy as np

from pamap2 import (
    BODY_PART_IDS,
    DATA_FILENAME,
    DATASET_ID,
    DEV_STARTS,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    LABEL_FILENAME,
    POSITION_ORDER,
    SEQ_LEN,
    TARGET_SR,
    WIN_SAMPLES_SRC,
    WIN_SAMPLES_TGT,
    ensure_dir,
    extract_device_sample,
    fill_nan_linear,
    iter_full_windows,
    resample_linear,
    robust_load_dat,
)


RAW_ID_TO_ACTIVITY_ID = {
    6: 19,    # cycling
    7: 211,   # Nordic walking
    9: 20,    # watching TV
    10: 21,   # computer work
    11: 22,   # car driving
    16: 23,   # vacuum cleaning
    17: 24,   # ironing
    18: 25,   # folding laundry
    19: 26,   # house cleaning
    20: 212,  # playing soccer
}
TARGET_RAW_IDS = list(RAW_ID_TO_ACTIVITY_ID.keys())
CONFIG_KEY = f"PAMAP2_OTHER_{TARGET_SR}_{SEQ_LEN}"


def process_pamap2_other(pamap_root, out_root):
    data_list = []
    label_list = []

    for sid, subject_num in enumerate(range(101, 110)):
        subject_files = [
            f"subject{subject_num}.dat",
            f"subject{subject_num}_o.dat",
        ]

        for fname in subject_files:
            path = os.path.join(pamap_root, fname)
            if not os.path.isfile(path):
                continue

            arr = robust_load_dat(path)

            act_col = arr[:, 1].astype(np.int32)
            mask_any = np.isin(act_col, TARGET_RAW_IDS)
            arr = arr[mask_any, :]
            act_col_filtered = arr[:, 1].astype(np.int32)

            for act_id_raw in TARGET_RAW_IDS:
                seg = arr[act_col_filtered == act_id_raw, :]
                if seg.shape[0] < WIN_SAMPLES_SRC:
                    continue

                for win in iter_full_windows(seg, WIN_SAMPLES_SRC):
                    win = fill_nan_linear(win)
                    win120 = resample_linear(win, WIN_SAMPLES_TGT)

                    for dev in POSITION_ORDER:
                        start = DEV_STARTS[dev]
                        samp = extract_device_sample(win120, start)
                        data_list.append(samp)

                        activity_id = RAW_ID_TO_ACTIVITY_ID[act_id_raw]
                        global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + sid
                        body_part_id = BODY_PART_IDS[dev]
                        label_list.append([[activity_id, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check file paths and filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)

    print(f"[INFO] Final: data={data.shape}, labels={labels.shape}")

    out_dir = os.path.join(out_root, "PAMAP2")
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)

    return data.shape[0], out_dir


def update_data_config(config_path, total_size):
    cfg = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            print("[WARN] data_config.json is invalid. Recreating.")

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

    print(f"[INFO] Updated data_config: {CONFIG_KEY} (size={total_size})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare PAMAP2 other activities to npy format (20Hz, 6x120)."
    )
    parser.add_argument(
        "--pamap_root",
        type=str,
        default=os.path.join("PAMAP2", "PAMAP2_Dataset", "Protocol"),
        help="Path containing subject101.dat ... subject109.dat",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset_other",
        help="Output root; will create dataset_other/PAMAP2",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_other", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_pamap2_other(args.pamap_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config path: {args.config_path}")
