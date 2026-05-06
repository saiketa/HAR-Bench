import os
import json
import numpy as np
from scipy.io import loadmat

from uschad import (
    BODY_PART_ID,
    DATA_FILENAME,
    DATASET_DIRNAME,
    DATASET_ID,
    DIMENSION,
    GLOBAL_SUBJECT_ID_OFFSET,
    LABEL_FILENAME,
    METADATA_FILENAME,
    SEQ_LEN,
    TARGET_SR,
    WIN_SRC,
    WIN_TGT,
    ensure_dir,
    extract_sensor_readings,
    extract_subject_metadata,
    fill_nan_linear,
    iter_full_windows,
    parse_activity_from_filename,
    parse_subject_id_from_dir,
    resample_linear,
)


CONFIG_KEY = f"{DATASET_DIRNAME}_OTHER_{TARGET_SR}_{SEQ_LEN}"

# USC-HAD other raw activity id -> unified benchmark activity id
USCHAD_OTHER_TO_BENCH = {
    11: 103,  # Elevator Up
    12: 104,  # Elevator Down
}

TARGET_ACT_IDS = sorted(USCHAD_OTHER_TO_BENCH.keys())


def process_uschad_other(uschad_root, out_root):
    data_list = []
    label_list = []
    subject_metadata = {}

    for subj_dir in sorted(os.listdir(uschad_root)):
        subj_path = os.path.join(uschad_root, subj_dir)
        if not os.path.isdir(subj_path):
            continue

        try:
            user_id = parse_subject_id_from_dir(subj_dir)
            global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id
        except ValueError:
            continue

        for fname in sorted(os.listdir(subj_path)):
            if not fname.lower().endswith(".mat"):
                continue

            try:
                act_id_raw, _trial = parse_activity_from_filename(fname)
            except ValueError:
                continue

            if act_id_raw not in TARGET_ACT_IDS:
                continue

            fpath = os.path.join(subj_path, fname)

            mat = loadmat(fpath, squeeze_me=True, struct_as_record=False)
            if global_subject_id not in subject_metadata:
                subject_metadata[global_subject_id] = {
                    "source_subject_id": user_id + 1,
                    "local_subject_index": user_id,
                    "source_subject": subj_dir,
                    **extract_subject_metadata(mat),
                }

            sr = extract_sensor_readings(mat)

            if sr.ndim != 2:
                raise ValueError(f"Unexpected sensor_readings ndim={sr.ndim} in {fpath}")
            if sr.shape[1] != 6:
                if sr.shape[0] == 6:
                    sr = sr.T
                else:
                    raise ValueError(f"Unexpected sensor_readings shape {sr.shape} in {fpath}")

            sr = fill_nan_linear(sr)

            for win in iter_full_windows(sr, WIN_SRC):
                win120 = resample_linear(win, WIN_TGT).astype(np.float32)
                act_new_id = USCHAD_OTHER_TO_BENCH[act_id_raw]

                data_list.append(win120)
                label_list.append([[act_new_id, global_subject_id, BODY_PART_ID, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check USC-HAD path/structure/activity filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)
    labels = np.array(label_list, dtype=np.int64)

    print(f"[INFO] Final: data={data.shape}, labels={labels.shape}")

    out_dir = os.path.join(out_root, DATASET_DIRNAME)
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)
    with open(os.path.join(out_dir, METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in sorted(subject_metadata.items())},
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("Activity classes contained:", np.unique(labels[:, 0, 0]).astype(int).tolist())
    print("Global subject ids contained:", np.unique(labels[:, 0, 1]).astype(int).tolist())
    print("Body part ids contained:", np.unique(labels[:, 0, 2]).astype(int).tolist())
    print("Dataset ids contained:", np.unique(labels[:, 0, 3]).astype(int).tolist())

    return data.shape[0], out_dir


def update_data_config(config_path, total_size):
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

    print(f"[INFO] Updated data_config: {CONFIG_KEY} (size={total_size})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare USC-HAD other activities to npy (20Hz, 6x120, unified labels)."
    )
    parser.add_argument(
        "--uschad_root",
        type=str,
        default="./USC-HAD",
        help="Path containing Subject1 ... Subject14 folders",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset_other",
        help="Output root; will create dataset_other/USC-HAD",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_other", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_uschad_other(args.uschad_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f" - {METADATA_FILENAME}")
    print(f"Config path: {args.config_path}")
