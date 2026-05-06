import os
import json
import numpy as np

# =========================
# 原始 PAMAP2 活动ID → 规范名
# =========================
RAW_ID_TO_CANON = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    12: "upstairs",
    13: "downstairs",
    24: "jumping",
}

# 统一活动顺序（最终标签顺序）
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

# 仅处理这些原始活动ID
TARGET_RAW_IDS = list(RAW_ID_TO_CANON.keys())

# 位置标签（从左到右）
POSITION_ORDER = ["hand", "chest", "ankle"]
BODY_PART_IDS = {
    "hand": 6,
    "chest": 0,
    "ankle": 5,
}

# 采样参数
SRC_SR = 100
TARGET_SR = 20
WIN_SEC = 6
WIN_SAMPLES_SRC = SRC_SR * WIN_SEC    # 600
WIN_SAMPLES_TGT = TARGET_SR * WIN_SEC # 120

# 列说明（0-based）
# 0: timestamp
# 1: activity_id
# 2: heart_rate
# 3..19: hand (17 cols)
# 20..36: chest (17 cols)
# 37..53: ankle (17 cols)
DEV_STARTS = {
    "hand": 3,
    "chest": 20,
    "ankle": 37,
}

# 每个 17 列设备块中保留：
# acc1_x, acc1_y, acc1_z, gyro_x, gyro_y, gyro_z
# 对应设备块内部 0-based 相对索引：
# [1,2,3] = 1号三轴加速度
# [7,8,9] = 三轴角速度
KEEP_WITHIN_BLOCK = [1, 2, 3, 7, 8, 9]

DIMENSION = 6
SEQ_LEN = WIN_SAMPLES_TGT
USER_SIZE = 9  # subject101..subject109
GLOBAL_SUBJECT_ID_OFFSET = 153
DATASET_ID = 6

CONFIG_KEY = f"PAMAP2_{TARGET_SR}_{SEQ_LEN}_{ACTIVITY_SIZE}activity"
DATA_FILENAME = "data.npy"
LABEL_FILENAME = "label.npy"


# =========================
# 工具函数
# =========================
def robust_load_dat(path):
    """
    读取 PAMAP2 .dat：空白分隔，可能含 NaN。
    返回 ndarray [T, 54]，dtype=float32
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 54:
        raise ValueError(f"Unexpected shape {arr.shape} for file: {path}")
    if arr.shape[1] > 54:
        arr = arr[:, :54]
    return arr.astype(np.float32)


def iter_full_windows(arr_rows_54, win_len):
    """
    将二维数组（T,54）按 win_len 无重叠切分，丢弃不足一窗的尾部。
    产出形状 (win_len, 54) 的窗口。
    """
    T = arr_rows_54.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr_rows_54[: n * win_len, :]
    for i in range(n):
        yield trunc[i * win_len : (i + 1) * win_len, :]


def fill_nan_linear(x):
    """
    对窗口 (L, D) 沿时间维对 NaN 做线性插值；
    全 NaN 列填 0；端点自动外延。
    """
    x = x.copy()
    L, D = x.shape
    idx = np.arange(L, dtype=np.float32)

    for d in range(D):
        col = x[:, d]
        mask = np.isnan(col)
        if not mask.any():
            continue
        if mask.all():
            x[:, d] = 0.0
            continue
        x[mask, d] = np.interp(idx[mask], idx[~mask], col[~mask])

    return x


def resample_linear(win_src, tgt_len):
    """
    将 (L_src, D) 线性插值到 (tgt_len, D)，假设等间隔采样。
    """
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def extract_device_sample(win_120x54, dev_start):
    """
    从重采样后的窗口 (120,54) 中抽取指定设备 hand/chest/ankle 的 6 列：
    acc1_x, acc1_y, acc1_z, gyro_x, gyro_y, gyro_z

    并进行单位换算：
    - 加速度: m/s² -> g
    - 角速度: rad/s -> dps
    """
    cols = [dev_start + k for k in KEEP_WITHIN_BLOCK]
    samp = win_120x54[:, cols].astype(np.float32)  # (120,6)

    # 前3列: 加速度 m/s² -> g
    samp[:, 0:3] /= 9.80665

    # 后3列: 角速度 rad/s -> dps
    samp[:, 3:6] *= (180.0 / np.pi)

    return samp


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# =========================
# 主流程
# =========================
def process_pamap2(pamap_root, out_root):
    """
    pamap_root: 包含 subject101.dat ... subject109.dat
    out_root: 输出目录根（将生成 out_root/PAMAP2/）
    """
    data_list = []
    label_list = []

    subjects = [f"subject{sid}.dat" for sid in range(101, 110)]

    for sid, fname in enumerate(subjects):  # sid: 0..8
        path = os.path.join(pamap_root, fname)
        if not os.path.isfile(path):
            print(f"[WARN] missing file: {path}, skip")
            continue

        arr = robust_load_dat(path)  # [T,54]

        # 只保留目标活动
        act_col = arr[:, 1].astype(np.int32)
        mask_any = np.isin(act_col, TARGET_RAW_IDS)
        arr = arr[mask_any, :]

        # 同步活动列，避免索引混淆
        act_col_filtered = arr[:, 1].astype(np.int32)

        # 按目标活动 ID 分组
        for act_id_raw in TARGET_RAW_IDS:
            seg = arr[act_col_filtered == act_id_raw, :]
            if seg.shape[0] < WIN_SAMPLES_SRC:
                continue  # 不足一窗

            # 6s 无重叠切窗
            for win in iter_full_windows(seg, WIN_SAMPLES_SRC):  # (600,54)
                # NaN 线性填充
                win = fill_nan_linear(win)

                # 重采样到 120×54
                win120 = resample_linear(win, WIN_SAMPLES_TGT)

                # 按 hand/chest/ankle 提取 3 个样本，并记录统一身体部位编号
                for dev in POSITION_ORDER:
                    start = DEV_STARTS[dev]
                    samp = extract_device_sample(win120, start)  # (120,6)
                    data_list.append(samp)

                    # 原始活动ID -> 规范名 -> 统一标签ID
                    act_name = RAW_ID_TO_CANON[act_id_raw]
                    act_new_id = ACTIVITY_TO_ID[act_name]

                    # label = [activity_id, global_subject_id, body_part_id, dataset_id]
                    global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + sid
                    body_part_id = BODY_PART_IDS[dev]
                    label_list.append([[act_new_id, global_subject_id, body_part_id, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check file paths and filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)  # [N,120,6]
    labels = np.array(label_list, dtype=np.int64)          # [N,1,4]

    print(f"[INFO] Final: data={data.shape}, labels={labels.shape}")

    out_dir = os.path.join(out_root, "PAMAP2")
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


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare PAMAP2 to npy format (20Hz, 6x120, 8 activities) with position labels."
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
        default="dataset_8",
        help="Output root; will create dataset_8/PAMAP2",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_8", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_pamap2(args.pamap_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config path: {args.config_path}")
