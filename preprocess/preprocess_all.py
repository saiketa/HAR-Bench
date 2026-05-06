#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_DIR = PROJECT_ROOT / "preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VERSION = "20_120"


@dataclass
class Job:
    name: str
    module: str
    split: str
    source: str
    runner: str
    optional: bool = False


def extract_zip_if_needed(zip_path, marker_path):
    zip_path = Path(zip_path)
    marker_path = Path(marker_path)
    if marker_path.exists() or not zip_path.exists():
        return
    print(f"[extract] {zip_path} -> {zip_path.parent}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)


def prepare_archives(datasets_root, selected=None):
    datasets_root = Path(datasets_root)
    selected = None if selected is None else {x.lower() for x in selected}

    def wants(name):
        return selected is None or name.lower() in selected

    if wants("HHAR"):
        extract_zip_if_needed(
            datasets_root / "HHAR" / "Activity recognition exp.zip",
            datasets_root / "HHAR" / "Activity recognition exp",
        )
    if wants("PAMAP2"):
        extract_zip_if_needed(
            datasets_root / "PAMAP2" / "PAMAP2_Dataset.zip",
            datasets_root / "PAMAP2" / "PAMAP2_Dataset" / "Protocol",
        )
    if wants("WISDM"):
        extract_zip_if_needed(
            datasets_root / "WISDM" / "wisdm-dataset.zip",
            datasets_root / "WISDM" / "wisdm-dataset" / "raw",
        )
    if wants("Motion"):
        extract_zip_if_needed(
            datasets_root / "Motion" / "B_Accelerometer_data.zip",
            datasets_root / "Motion" / "B_Accelerometer_data",
        )
        extract_zip_if_needed(
            datasets_root / "Motion" / "C_Gyroscope_data.zip",
            datasets_root / "Motion" / "C_Gyroscope_data",
        )


def import_module(name):
    return importlib.import_module(name)


def update_config(module, config_path, result):
    if not hasattr(module, "update_data_config"):
        return
    if isinstance(result, tuple) and len(result) >= 2 and hasattr(result[0], "shape") and hasattr(result[1], "shape"):
        module.update_data_config(config_path=str(config_path), version=VERSION, data=result[0], label=result[1])
        return
    if isinstance(result, tuple) and len(result) >= 1:
        total = result[0]
        if len(result) >= 3 and isinstance(result[2], int):
            module.update_data_config(str(config_path), total, result[2])
        else:
            module.update_data_config(str(config_path), total)


def run_job(job, datasets_root, data_root, data_other_root, dry_run=False):
    source = Path(datasets_root) / job.source
    out_base = Path(data_root if job.split == "main" else data_other_root)
    config_path = out_base / "data_config.json"

    if not source.exists():
        zip_sources = {
            "HHAR": Path(datasets_root) / "HHAR" / "Activity recognition exp.zip",
            "PAMAP2": Path(datasets_root) / "PAMAP2" / "PAMAP2_Dataset.zip",
            "WISDM": Path(datasets_root) / "WISDM" / "wisdm-dataset.zip",
        }
        if dry_run and job.name in zip_sources and zip_sources[job.name].exists():
            print(f"[dry-run] {job.name}: source will be created by extracting {zip_sources[job.name]}")
        elif job.optional:
            msg = f"[skip] {job.name}: source not found: {source}"
            print(msg)
            return "skipped"
        else:
            raise FileNotFoundError(f"[skip] {job.name}: source not found: {source}")

    print(f"\n=== {job.name} ({job.split}) ===")
    print(f"source: {source}")
    print(f"output: {out_base}")

    if dry_run:
        return "dry-run"

    module = import_module(job.module)

    if job.runner == "root_dsads":
        result = module.process_dsads(str(source), str(out_base))
    elif job.runner == "root_dsads_other":
        result = module.process_dsads_other(str(source), str(out_base))
    elif job.runner == "root_pamap2":
        result = module.process_pamap2(str(source), str(out_base))
    elif job.runner == "root_pamap2_other":
        result = module.process_pamap2_other(str(source), str(out_base))
    elif job.runner == "root_kuhar":
        result = module.process_kuhar(str(source), str(out_base))
    elif job.runner == "root_kuhar_other":
        result = module.process_kuhar_other(str(source), str(out_base))
    elif job.runner == "root_uschad":
        result = module.process_uschad(str(source), str(out_base))
    elif job.runner == "root_uschad_other":
        result = module.process_uschad_other(str(source), str(out_base))
    elif job.runner == "final_harsense":
        result = module.process_harsense(str(source), str(out_base / "HARSense"), VERSION)
    elif job.runner == "final_hhar":
        result = module.preprocess_hhar(str(source), str(out_base / "HHAR"), VERSION, str(config_path), window_time=50, seq_len=120, jump=0)
    elif job.runner == "final_hhar_other":
        result = module.preprocess_hhar_other(str(source), str(out_base / "HHAR"), VERSION, str(config_path), window_time=50, seq_len=120, jump=0)
    elif job.runner == "final_mhealth":
        result = module.process_mhealth(str(source), str(out_base / "MHEALTH"), VERSION)
    elif job.runner == "final_mhealth_other":
        result = module.process_mhealth_other(str(source), str(out_base / "MHEALTH"), VERSION)
    elif job.runner == "final_motion":
        result = module.preprocess(str(source), str(out_base / "Motion"), VERSION, target_window=50, seq_len=120)
    elif job.runner == "final_realworld":
        result = module.process_realworld(str(source), str(out_base / "RealWorld"), VERSION)
    elif job.runner == "final_shoaib":
        result = module.preprocess(str(source), str(out_base / "Shoaib"), VERSION, target_window=50, seq_len=120, position_num=5)
    elif job.runner == "final_shoaib_other":
        result = module.preprocess(str(source), str(out_base / "Shoaib"), VERSION, target_window=50, seq_len=120, position_num=5)
    elif job.runner == "final_tnda":
        result = module.process_tnda_har(str(source), str(out_base / "TNDA-HAR"), VERSION)
    elif job.runner == "final_tnda_other":
        result = module.process_tnda_har_other(str(source), str(out_base / "TNDA-HAR"), VERSION)
    elif job.runner == "final_uci":
        result = module.preprocess(str(source), str(out_base / "UCI"), VERSION, raw_sr=50, target_sr=20, seq_len=120)
    elif job.runner == "final_uci_other":
        result = module.preprocess(str(source), str(out_base / "UCI"), VERSION, raw_sr=50, target_sr=20, seq_len=120)
    elif job.runner == "final_ut":
        result = module.process_ut_complex(str(source), str(out_base / "UT-Complex"), VERSION)
    elif job.runner == "final_ut_other":
        result = module.process_ut_complex_other(str(source), str(out_base / "UT-Complex"), VERSION)
    elif job.runner == "final_wisdm":
        result = module.process_wisdm(str(source), str(out_base / "WISDM"), VERSION)
    elif job.runner == "final_wisdm_other":
        result = module.process_wisdm_other(str(source), str(out_base / "WISDM"), VERSION)
    else:
        raise ValueError(f"Unknown runner: {job.runner}")

    update_config(module, config_path, result)
    return "ok"


MAIN_JOBS = [
    Job("DSADS", "dsads", "main", "DSADS", "root_dsads"),
    Job("HARSense", "harsense", "main", "HARSense", "final_harsense"),
    Job("HHAR", "hhar", "main", "HHAR/Activity recognition exp", "final_hhar"),
    Job("KU-HAR", "kuhar", "main", "KU-HAR/1.Raw_time_domian_data", "root_kuhar"),
    Job("MHEALTH", "mhealth", "main", "Mhealth", "final_mhealth"),
    Job("Motion", "motion", "main", "Motion", "final_motion"),
    Job("PAMAP2", "pamap2", "main", "PAMAP2/PAMAP2_Dataset/Protocol", "root_pamap2"),
    Job("RealWorld", "realworld", "main", "RealWorld", "final_realworld"),
    Job("Shoaib", "shoaib", "main", "Shoaib/DataSet", "final_shoaib"),
    Job("TNDA-HAR", "tnda_har", "main", "TNDA-HAR", "final_tnda"),
    Job("UCI", "uci", "main", "UCI/RawData", "final_uci"),
    Job("USC-HAD", "uschad", "main", "USC-HAD", "root_uschad"),
    Job("UT-Complex", "ut_complex", "main", "UT-Complex/UT_Data_Complex", "final_ut"),
    Job("WISDM", "wisdm", "main", "WISDM/wisdm-dataset/raw", "final_wisdm"),
]

OTHER_JOBS = [
    Job("DSADS", "dsads_other", "other", "DSADS", "root_dsads_other"),
    Job("HHAR", "hhar_other", "other", "HHAR/Activity recognition exp", "final_hhar_other", optional=True),
    Job("KU-HAR", "kuhar_other", "other", "KU-HAR/1.Raw_time_domian_data", "root_kuhar_other"),
    Job("MHEALTH", "mhealth_other", "other", "Mhealth", "final_mhealth_other"),
    Job("PAMAP2", "pamap2_other", "other", "PAMAP2/PAMAP2_Dataset/Protocol", "root_pamap2_other"),
    Job("Shoaib", "shoaib_other", "other", "Shoaib/DataSet", "final_shoaib_other"),
    Job("TNDA-HAR", "tnda_har_other", "other", "TNDA-HAR", "final_tnda_other"),
    Job("UCI", "uci_other", "other", "UCI/RawData", "final_uci_other"),
    Job("USC-HAD", "uschad_other", "other", "USC-HAD", "root_uschad_other"),
    Job("UT-Complex", "ut_complex_other", "other", "UT-Complex/UT_Data_Complex", "final_ut_other"),
    Job("WISDM", "wisdm_other", "other", "WISDM/wisdm-dataset/raw", "final_wisdm_other"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess HAR-Bench raw datasets into data/ and data_other/.")
    parser.add_argument("--datasets_root", type=str, default="./datasets")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--data_other_root", type=str, default="./data_other")
    parser.add_argument("--only", nargs="+", default=None, help="Run only selected dataset names, e.g. DSADS WISDM")
    parser.add_argument("--main_only", action="store_true", help="Generate only data/")
    parser.add_argument("--other_only", action="store_true", help="Generate only data_other/")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.main_only and args.other_only:
        raise ValueError("--main_only and --other_only are mutually exclusive")

    selected = None if args.only is None else {x.lower() for x in args.only}
    if args.dry_run:
        print("[dry-run] Archive extraction is skipped.")
    else:
        prepare_archives(args.datasets_root, selected=selected)
    jobs = []
    if not args.other_only:
        jobs.extend(MAIN_JOBS)
    if not args.main_only:
        jobs.extend(OTHER_JOBS)

    failures = []
    for job in jobs:
        if selected is not None and job.name.lower() not in selected:
            continue
        try:
            status = run_job(job, args.datasets_root, args.data_root, args.data_other_root, dry_run=args.dry_run)
            print(f"[{status}] {job.name} ({job.split})")
        except Exception as exc:
            print(f"[failed] {job.name} ({job.split}): {type(exc).__name__}: {exc}")
            failures.append((job.name, job.split, exc))

    if failures:
        print("\nFailures:")
        for name, split, exc in failures:
            print(f" - {name} ({split}): {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("\n[DONE] Preprocessing finished.")


if __name__ == "__main__":
    main()
