import os
import re
import json
import numpy as np
from scipy.io import loadmat

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
# USC-HAD 原始活动号 -> 规范名
# 注意：USC-HAD 原始 10 是 laying，这里统一规范为 lying
# 注意：USC-HAD 原始 7 是 Jumping Up
# =========================
ACTID_TO_CANON = {
    1: "walking",
    4: "upstairs",
    5: "downstairs",
    6: "running",
    7: "jumping",
    8: "sitting",
    9: "standing",
    10: "lying",
}
TARGET_ACT_IDS = [1, 4, 5, 6, 7, 8, 9, 10]

SRC_SR = 100
TARGET_SR = 20
WIN_SEC = 6
WIN_SRC = SRC_SR * WIN_SEC    # 600
WIN_TGT = TARGET_SR * WIN_SEC # 120

DIMENSION = 6
SEQ_LEN = WIN_TGT
USER_SIZE = 14  # Subject1..Subject14
GLOBAL_SUBJECT_ID_OFFSET = 267
BODY_PART_ID = 7
DATASET_ID = 11

DATASET_DIRNAME = "USC-HAD"
CONFIG_KEY = f"{DATASET_DIRNAME}_{TARGET_SR}_{SEQ_LEN}"
DATA_FILENAME = "data.npy"
LABEL_FILENAME = "label.npy"
METADATA_FILENAME = "subject_metadata.json"


# =========================
# 工具函数
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_subject_id_from_dir(dirname):
    """
    Subject1 -> 0, ..., Subject14 -> 13 local id.
    """
    m = re.search(r"Subject\s*(\d+)", dirname, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse subject id from dirname: {dirname}")
    sid = int(m.group(1))
    if not (1 <= sid <= 14):
        raise ValueError(f"Subject id out of range: {sid} in {dirname}")
    return sid - 1


def parse_activity_from_filename(fname):
    """
    从 a1t1.mat / a4t4.mat / a12t5.mat 解析 activity_id 和 trial_id
    """
    m = re.search(r"a(\d+)\s*t(\d+)", fname, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 容错：若 trial 解析不到，至少要拿到 activity id
    m2 = re.search(r"a(\d+)", fname, re.IGNORECASE)
    if m2:
        return int(m2.group(1)), None

    raise ValueError(f"Cannot parse activity id from filename: {fname}")


def fill_nan_linear(x):
    """
    对 (L,D) 沿时间维线性插值填充 NaN；全 NaN 列置 0。
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
    将 (L_src, D) 线性插值为 (tgt_len, D)，假设等间隔采样。
    """
    L_src = win_src.shape[0]
    t_src = np.linspace(0.0, 1.0, L_src, endpoint=False, dtype=np.float32)
    t_tgt = np.linspace(0.0, 1.0, tgt_len, endpoint=False, dtype=np.float32)

    out = np.empty((tgt_len, win_src.shape[1]), dtype=np.float32)
    for d in range(win_src.shape[1]):
        out[:, d] = np.interp(t_tgt, t_src, win_src[:, d])

    return out


def iter_full_windows(arr, win_len):
    """
    将 (T,D) 按 win_len 无重叠切窗，丢弃尾部不足窗的数据。
    """
    T = arr.shape[0]
    n = T // win_len
    if n == 0:
        return
    trunc = arr[: n * win_len, :]
    for i in range(n):
        yield trunc[i * win_len : (i + 1) * win_len, :]


def normalize_field_name(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def extract_sensor_readings(matdict):
    """
    从 .mat 字典中抽取 sensor_readings，期望得到 (T,6)
    兼容：
    - 顶层变量名 sensor_readings
    - 顶层 struct 中的 sensor_readings 字段
    """
    # 1) 顶层直接变量
    for k, v in matdict.items():
        if k.startswith("__"):
            continue
        if normalize_field_name(k) == "sensor_readings":
            arr = np.array(v)
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                if arr.shape[1] == 6:
                    return arr.astype(np.float32)
                if arr.shape[0] == 6:
                    return arr.T.astype(np.float32)

    # 2) struct 内部字段
    def try_get_sr(obj):
        try:
            sr = getattr(obj, "sensor_readings", None)
            if sr is None:
                for attr in dir(obj):
                    if normalize_field_name(attr) == "sensor_readings":
                        sr = getattr(obj, attr)
                        break
            if sr is None:
                return None
            arr = np.array(sr)
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                return None
            if arr.shape[1] == 6:
                return arr.astype(np.float32)
            if arr.shape[0] == 6:
                return arr.T.astype(np.float32)
        except Exception:
            return None
        return None

    for k, v in matdict.items():
        if k.startswith("__"):
            continue

        sr = try_get_sr(v)
        if sr is not None:
            return sr

        if isinstance(v, np.ndarray) and v.dtype == np.object_:
            for item in v.ravel():
                sr = try_get_sr(item)
                if sr is not None:
                    return sr

    raise KeyError("sensor_readings not found in .mat file")


def scalar_to_python(value):
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def extract_mat_field(matdict, target_name):
    """
    从 .mat 顶层或顶层 struct 中抽取标量字段，如 age / height / weight。
    """
    target = normalize_field_name(target_name)

    for k, v in matdict.items():
        if k.startswith("__"):
            continue
        if normalize_field_name(k) == target:
            return scalar_to_python(v)

    def try_get_field(obj):
        try:
            value = getattr(obj, target_name, None)
            if value is None:
                for attr in dir(obj):
                    if normalize_field_name(attr) == target:
                        value = getattr(obj, attr)
                        break
            if value is None:
                return None
            return scalar_to_python(value)
        except Exception:
            return None

    for k, v in matdict.items():
        if k.startswith("__"):
            continue

        value = try_get_field(v)
        if value is not None:
            return value

        if isinstance(v, np.ndarray) and v.dtype == np.object_:
            for item in v.ravel():
                value = try_get_field(item)
                if value is not None:
                    return value

    return None


def extract_subject_metadata(matdict):
    return {
        "age": extract_mat_field(matdict, "age"),
        "height": extract_mat_field(matdict, "height"),
        "weight": extract_mat_field(matdict, "weight"),
    }


# =========================
# 主流程
# =========================
def process_uschad(uschad_root, out_root):
    """
    uschad_root:
      └─ Subject1 ... Subject14
           └─ a1t1.mat, a4t4.mat, ...
    """
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

        # 每个受试者的各活动 trial 分开处理
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

            # 读取 .mat
            mat = loadmat(fpath, squeeze_me=True, struct_as_record=False)
            if global_subject_id not in subject_metadata:
                subject_metadata[global_subject_id] = {
                    "source_subject_id": user_id + 1,
                    "local_subject_index": user_id,
                    "source_subject": subj_dir,
                    **extract_subject_metadata(mat),
                }

            sr = extract_sensor_readings(mat)  # (T,6)

            if sr.ndim != 2:
                raise ValueError(f"Unexpected sensor_readings ndim={sr.ndim} in {fpath}")
            if sr.shape[1] != 6:
                if sr.shape[0] == 6:
                    sr = sr.T
                else:
                    raise ValueError(f"Unexpected sensor_readings shape {sr.shape} in {fpath}")

            # 缺失值处理
            sr = fill_nan_linear(sr)

            # 6s 无重叠切窗（100Hz -> 600）
            for win in iter_full_windows(sr, WIN_SRC):  # (600,6)
                # 重采样到 20Hz -> (120,6)
                win120 = resample_linear(win, WIN_TGT).astype(np.float32)

                act_name = ACTID_TO_CANON[act_id_raw]
                act_new_id = ACTIVITY_TO_ID[act_name]

                data_list.append(win120)
                label_list.append([[act_new_id, global_subject_id, BODY_PART_ID, DATASET_ID]])

    if not data_list:
        raise RuntimeError("No samples produced. Check USC-HAD path/structure/activity filters.")

    data = np.stack(data_list, axis=0).astype(np.float32)  # [N,120,6]
    labels = np.array(label_list, dtype=np.int64)          # [N,1,4]

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


# =========================
# 主入口
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare USC-HAD to npy (20Hz, 6x120, 8 activities, unified label order)."
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
        default="dataset_8",
        help="Output root; will create dataset_8/USC-HAD",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join("dataset_8", "data_config.json"),
        help="Path to data_config.json to update/create",
    )
    args = parser.parse_args()

    total, out_dir = process_uschad(args.uschad_root, args.dataset_root)
    update_data_config(args.config_path, total)

    print("[DONE]")
    print(f"Saved to: {out_dir}")
    print(f" - {DATA_FILENAME}")
    print(f" - {LABEL_FILENAME}")
    print(f" - {METADATA_FILENAME}")
    print(f"Config path: {args.config_path}")
