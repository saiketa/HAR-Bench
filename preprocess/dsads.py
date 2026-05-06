import os
import json
import numpy as np
import re

# =========================
# 全局统一活动顺序（8类）
# =========================
ACTIVITY_ORDER = [
    "sitting",
    "standing",
    "lying",
    "upstairs",
    "downstairs",
    "walking",
    "running",
    "jumping",
]
ACTIVITY_TO_ID = {name: i for i, name in enumerate(ACTIVITY_ORDER)}
ACTIVITY_SIZE = len(ACTIVITY_ORDER)

# =========================
# DSADS 原始活动文件夹 -> 规范名
# 注意：DSADS a18 对应 jumping
# =========================
ACTIVITY_FOLDERS = {
    "a01": "sitting",
    "a02": "standing",
    "a03": "lying",
    "a05": "upstairs",
    "a06": "downstairs",
    "a09": "walking",
    "a12": "running",
    "a18": "jumping",
}

# =========================
# DSADS 5组传感器从左到右，对应统一身体部位编号
# =========================
POSITION_ORDER = ["torso", "right_arm", "left_arm", "right_leg", "left_leg"]
BODY_PART_IDS = [0, 1, 1, 3, 3]
BODY_PART_LABEL = [
    "chest",
    "wrist",
    "thigh",
    "knee",
    "shin",
    "ankle",
    "handheld",
    "trouser_pocket",
]
BODY_PART_SIZE = len(BODY_PART_LABEL)

# DSADS 数据集统一编号
DATASET_ID = 0

# =========================
# 采样设置
# =========================
SRC_SR = 25
TARGET_SR = 20
WIN_SEC = 6
WIN_SAMPLES_SRC = SRC_SR * WIN_SEC     # 150
WIN_SAMPLES_TGT = TARGET_SR * WIN_SEC  # 120

# =========================
# 列布局
# 45列 = 5组 × 9轴
# 每组顺序：
# acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
# =========================
GROUPS = 5
AXES_PER_GROUP = 9
KEEP_IDXS_WITHIN_GROUP = [0, 1, 2, 3, 4, 5]  # 仅保留 acc(3) + gyro(3)

# =========================
# 输出配置
# =========================
DIMENSION = 6
SEQ_LEN = WIN_SAMPLES_TGT
USER_SIZE = 8  # p1..p8

CONFIG_KEY = f"DSADS_{TARGET_SR}_{SEQ_LEN}"
DATA_FILENAME = "data.npy"
LABEL_FILENAME = "label.npy"


# =========================
# 基础函数
# =========================
def robust_read_txt_125x45(path):
    """
    读取 DSADS 单个 txt -> (125,45) float32
    兼容逗号/分号/空白分隔，自动裁到标准形状。
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            toks = [t for t in re.split(r"[,\s;]+", line) if t != ""]
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                raise ValueError(f"Failed to parse numeric row in {path}: {line[:120]}")

    arr = np.asarray(rows, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected ndim {arr.ndim} in {path}")
    if arr.shape[1] < 45:
        raise ValueError(f"Got {arr.shape[1]} columns (<45) in {path}")
    if arr.shape[1] > 45:
        arr = arr[:, :45]

    if arr.shape[0] < 125:
        raise ValueError(f"Got {arr.shape[0]} rows (<125) in {path}")
    if arr.shape[0] > 125:
        arr = arr[:125, :]

    return arr  # (125,45)


def concat_subject_activity(txt_files):
    """
    将 s01..s60（每个 125×45）在时间上拼接成 7500×45
    """
    mats = [robust_read_txt_125x45(f) for f in txt_files]
    cat = np.vstack(mats)
    if cat.shape[0] != 60 * 125:
        raise ValueError(f"Concatenation length mismatch: {cat.shape}")
    return cat.astype(np.float32)


def split_to_windows_6s(arr_7500x45):
    """
    无重叠 6 秒窗：7500 / 150 = 50 窗，得到 [50,150,45]
    """
    total = arr_7500x45.shape[0]
    n_windows = total // WIN_SAMPLES_SRC
    if n_windows == 0:
        return np.empty((0, WIN_SAMPLES_SRC, arr_7500x45.shape[1]), dtype=np.float32)
    arr = arr_7500x45[: n_windows * WIN_SAMPLES_SRC, :]
    arr = arr.reshape(n_windows, WIN_SAMPLES_SRC, arr.shape[1]).astype(np.float32)
    return arr


def resample_150_to_120_linear(win_150x45):
    """
    线性插值将 [150,45] -> [120,45]
    """
    t_src = np.linspace(0.0, 1.0, WIN_SAMPLES_SRC, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, WIN_SAMPLES_TGT, endpoint=False, dtype=np.float32)

    out = np.empty((WIN_SAMPLES_TGT, win_150x45.shape[1]), dtype=np.float32)
    for c in range(win_150x45.shape[1]):
        out[:, c] = np.interp(t_tgt, t_src, win_150x45[:, c])

    return out


def split_45cols_to_5x6(win_120x45):
    """
    将 45 列拆为 5 组，每组 9 轴，取其中前 6 轴（acc+gyro），得到 5 个 [120,6]
    """
    samples = []
    for g in range(GROUPS):
        start = g * AXES_PER_GROUP
        cols9 = win_120x45[:, start : start + AXES_PER_GROUP]
        cols6 = cols9[:, KEEP_IDXS_WITHIN_GROUP]
        samples.append(cols6.astype(np.float32))
    return samples


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# 主流程
# =========================
def process_dsads(dsads_root, out_root):
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

    # 按统一活动顺序组织遍历，但只处理 DSADS 实际存在的活动
    activity_folders_sorted = []
    for act_name in ACTIVITY_ORDER:
        candidates = [k for k, v in ACTIVITY_FOLDERS.items() if v == act_name]
        if candidates:
            activity_folders_sorted.append(candidates[0])

    for a_folder in activity_folders_sorted:
        act_name = ACTIVITY_FOLDERS[a_folder]
        act_id = ACTIVITY_TO_ID[act_name]  # 使用全局统一编号
        a_path = os.path.join(dsads_root, a_folder)

        if not os.path.isdir(a_path):
            print(f"[WARN] Activity folder not found: {a_path}, skip")
            continue

        # subjects p1..p8
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

            # s01..s60
            txts = [os.path.join(s_path, f"s{idx:02d}.txt") for idx in range(1, 61)]
            missing = [t for t in txts if not os.path.isfile(t)]
            if missing:
                print(f"[WARN] Missing {len(missing)} files under {s_path}, skip this subject")
                continue

            # 拼接 -> [7500,45]
            arr = concat_subject_activity(txts)

            # 切窗 -> [50,150,45]
            wins = split_to_windows_6s(arr)

            # 每窗重采样到 [120,45]，再拆成 5 个 [120,6]
            for w in range(wins.shape[0]):
                win120x45 = resample_150_to_120_linear(wins[w])
                five_samples = split_45cols_to_5x6(win120x45)

                for pos_id, samp in enumerate(five_samples):
                    data_list.append(samp)  # [120,6]
                    body_part_id = BODY_PART_IDS[pos_id]
                    global_subject_id = subj_idx
                    label_list.append([[act_id, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples were produced. Please check DSADS path and structure.")

    data = np.stack(data_list, axis=0).astype(np.float32)   # [N,120,6]
    labels = np.array(label_list, dtype=np.int64)           # [N,1,4]

    print(f"[INFO] Final dataset shape: data={data.shape}, labels={labels.shape}")

    out_dir = os.path.join(out_root, "DSADS")
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)

    return data.shape[0], out_dir


# =========================
# 更新 data_config.json
# =========================
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


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare DSADS to npy format (20Hz, 6x120, 8 activities) with position labels."
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
        default="dataset_8",
        help="Output root; will create dataset_8/DSADS",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_8", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_dsads(args.dsads_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved data to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config updated at: {args.config_path}")
