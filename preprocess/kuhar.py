import os
import csv
import json
import numpy as np

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
# KU-HAR 动作文件夹 -> 规范名
# =========================
FOLDER_TO_CANON = {
    "0.Stand": "standing",
    "1.Sit": "sitting",
    "5.Lay": "lying",
    "8.Jump": "jumping",
    "11.Walk": "walking",
    "14.Run": "running",
    "15.Stair-up": "upstairs",
    "16.Stair-down": "downstairs",
}
TARGET_FOLDERS = set(FOLDER_TO_CANON.keys())

# =========================
# 采样参数
# =========================
SRC_SR = 100
TARGET_SR = 20
WIN_SEC = 6
WIN_SAMPLES_SRC = SRC_SR * WIN_SEC    # 600
WIN_SAMPLES_TGT = TARGET_SR * WIN_SEC # 120

# =========================
# 输出配置
# =========================
DIMENSION = 6
SEQ_LEN = WIN_SAMPLES_TGT
DATASET_DIRNAME = "KU-HAR"
GLOBAL_SUBJECT_ID_OFFSET = 29
BODY_PART_ID = 8
DATASET_ID = 3

CONFIG_KEY = f"{DATASET_DIRNAME}_{TARGET_SR}_{SEQ_LEN}_{ACTIVITY_SIZE}activity"
DATA_FILENAME = "data.npy"
LABEL_FILENAME = "label.npy"


# =========================
# 工具函数
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_base_userid(fname_no_ext: str) -> str:
    """
    从文件名第一个 '_' 之前提取原始用户ID
    例：
        1006_A_1 -> 1006
        1090_U_2 -> 1090
    """
    return fname_no_ext.split("_", 1)[0].strip()


def robust_read_csv_8(path):
    """
    读取 KU-HAR 单个 csv -> ndarray [T, 8] float32
    列顺序：
      [acc_ts, ax, ay, az, gyro_ts, gx, gy, gz]

    兼容：
    - 表头
    - 空行
    - 多余空格
    - 某些异常分号分隔场景
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        for i, line in enumerate(r):
            if not line:
                continue

            line = [x.strip() for x in line]
            if all(x == "" for x in line):
                continue

            try:
                vals = [float(x) for x in line]
            except ValueError:
                if i == 0:
                    # 可能是表头
                    continue
                else:
                    try:
                        vals = [float(x) for x in line[0].split(";")]
                    except Exception as e:
                        raise ValueError(f"Bad numeric row in {path} line {i+1}: {line}") from e

            if len(vals) < 8:
                continue
            rows.append(vals[:8])

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"Unexpected shape {arr.shape} in {path}")

    return arr  # [T,8]


def iter_full_windows(arr, win_len):
    """
    将 (T,D) 无重叠切窗，丢弃尾部不足一窗的数据
    """
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[: n * win_len, :]
    for i in range(n):
        yield trunc[i * win_len : (i + 1) * win_len, :]


def fill_nan_linear(x):
    """
    对 (L,D) 沿时间维做 NaN 线性插值；
    若整列都是 NaN，则置 0。
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
    将 (L_src, D) 线性插值到 (tgt_len, D)
    这里按样本点均匀插值，不直接使用时间戳。
    """
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def drop_timestamps_and_convert_units(win_120x8):
    """
    输入 (120,8):
        [acc_ts, ax, ay, az, gyro_ts, gx, gy, gz]

    输出 (120,6):
        [ax, ay, az, gx, gy, gz]

    单位换算：
    - 加速度: m/s² -> g
    - 角速度: rad/s -> dps
    """
    out = np.concatenate(
        [win_120x8[:, 1:4], win_120x8[:, 5:8]],
        axis=1
    ).astype(np.float32)

    # acc: m/s² -> g
    out[:, 0:3] /= 9.80665

    # gyro: rad/s -> dps
    out[:, 3:6] *= (180.0 / np.pi)

    return out


def collect_all_users(kuhar_root):
    """
    第一遍扫描：
    收集所有目标动作文件夹中出现过的原始用户ID，
    并构建稳定的 user_map。

    排序策略：
    - 能转成整数的按数值排序
    - 其余按字符串排序
    """
    numeric_ids = []
    string_ids = []

    for folder in sorted(os.listdir(kuhar_root)):
        if folder not in TARGET_FOLDERS:
            continue

        fdir = os.path.join(kuhar_root, folder)
        if not os.path.isdir(fdir):
            continue

        for fname in sorted(os.listdir(fdir)):
            if not fname.lower().endswith(".csv"):
                continue

            base = parse_base_userid(os.path.splitext(fname)[0])

            try:
                numeric_ids.append((int(base), base))
            except ValueError:
                string_ids.append(base)

    numeric_ids_sorted = [b for _, b in sorted(numeric_ids)]
    string_ids_sorted = sorted(set(string_ids))

    # 去重并保序
    all_ids = list(dict.fromkeys(numeric_ids_sorted + string_ids_sorted))
    user_map = {uid: i for i, uid in enumerate(all_ids)}

    return user_map


# =========================
# 主流程
# =========================
def process_kuhar(kuhar_root, out_root):
    """
    kuhar_root:
      ├─ 0.Stand/
      ├─ 1.Sit/
      ├─ 5.Lay/
      ├─ 8.Jump/
      ├─ 11.Walk/
      ├─ 14.Run/
      ├─ 15.Stair-up/
      └─ 16.Stair-down/
          └─ *.csv
    """
    # 第一遍：稳定用户映射
    user_map = collect_all_users(kuhar_root)
    print(f"[INFO] Collected {len(user_map)} unique users.")

    data_list = []
    label_list = []

    # 第二遍：正式处理数据
    for folder in sorted(os.listdir(kuhar_root)):
        if folder not in TARGET_FOLDERS:
            continue

        act_name = FOLDER_TO_CANON[folder]
        act_id = ACTIVITY_TO_ID[act_name]

        fdir = os.path.join(kuhar_root, folder)
        if not os.path.isdir(fdir):
            continue

        for fname in sorted(os.listdir(fdir)):
            if not fname.lower().endswith(".csv"):
                continue

            fpath = os.path.join(fdir, fname)
            arr = robust_read_csv_8(fpath)  # [T,8]

            base_uid = parse_base_userid(os.path.splitext(fname)[0])
            if base_uid not in user_map:
                # 理论上不会发生，这里做兜底
                user_map[base_uid] = len(user_map)
            user_id = user_map[base_uid]
            global_subject_id = GLOBAL_SUBJECT_ID_OFFSET + user_id

            if arr.shape[0] < WIN_SAMPLES_SRC:
                continue

            # 6秒无重叠切窗
            for win in iter_full_windows(arr, WIN_SAMPLES_SRC):  # (600,8)
                win = fill_nan_linear(win)

                # 重采样到 20Hz -> (120,8)
                win120 = resample_linear(win, WIN_SAMPLES_TGT)

                # 去时间戳并做单位换算 -> (120,6)
                samp = drop_timestamps_and_convert_units(win120)

                data_list.append(samp)
                label_list.append([[act_id, global_subject_id, BODY_PART_ID, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check KU-HAR path/folder names/CSV format.")

    data = np.stack(data_list, axis=0).astype(np.float32)  # [N,120,6]
    labels = np.array(label_list, dtype=np.int64)          # [N,1,4]

    print(f"[INFO] Final: data={data.shape}, labels={labels.shape}, users={len(user_map)}")

    out_dir = os.path.join(out_root, DATASET_DIRNAME)
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, DATA_FILENAME), data)
    np.save(os.path.join(out_dir, LABEL_FILENAME), labels)

    return data.shape[0], out_dir, len(user_map)


# =========================
# 更新 data_config.json
# =========================
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


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare KU-HAR to npy (20Hz, 6x120, 8 activities, unified label order)."
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
        default="dataset_8",
        help="Output root; will create dataset_8/KU-HAR",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_8", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir, user_size = process_kuhar(args.kuhar_root, args.dataset_root)
    update_data_config(args.config_path, total, user_size)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f"Config path: {args.config_path}")
