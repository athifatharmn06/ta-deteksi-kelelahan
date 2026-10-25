"""
exp12_decay_sensitivity.py
Sweep accumulator decay/gain configs with fixed weights.
Evaluates fatigue_alert events vs GT (DMD) and FP/session on YawDD Talking.
"""
import subprocess
import sys
import os
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Configuration ----
WEIGHTS = (0.45, 0.25, 0.30)
MAR_THRESH = 0.26
EAR_THRESH = 0.16

DECAY_CONFIGS = [
    {"name": "Base", "override": None},
    {"name": "SlowDecay", "override": {
        "params": {
            "mouth_decay_per_sec": 0.10,
            "head_decay_per_sec": 0.12,
            "eye_decay_per_sec": 0.08,
            "microsleep_decay_per_sec": 0.03,
        },
        "decay_scale_active": 0.45,
    }},
    {"name": "FastDecay", "override": {
        "params": {
            "mouth_decay_per_sec": 0.25,
            "head_decay_per_sec": 0.30,
            "eye_decay_per_sec": 0.18,
            "microsleep_decay_per_sec": 0.08,
        },
        "decay_scale_active": 0.65,
    }},
    {"name": "HighEventGain", "override": {
        "params": {
            "mouth_event_gain": 30.0,
            "head_event_gain": 28.0,
            "microsleep_event_gain": 50.0,
        }
    }},
    {"name": "LowEventGain", "override": {
        "params": {
            "mouth_event_gain": 18.0,
            "head_event_gain": 16.0,
            "microsleep_event_gain": 30.0,
        }
    }},
    {"name": "HighHoldGain", "override": {
        "params": {
            "head_hold_gain_per_sec": 4.0,
            "eye_hold_gain_per_sec": 3.2,
            "perclos_gain_per_sec": 1.2,
            "microsleep_hold_gain_per_sec": 10.0,
        }
    }},
    {"name": "LowHoldGain", "override": {
        "params": {
            "head_hold_gain_per_sec": 2.0,
            "eye_hold_gain_per_sec": 1.8,
            "perclos_gain_per_sec": 0.6,
            "microsleep_hold_gain_per_sec": 5.0,
        }
    }},
]

DS_MAPPING = {
    1:  {"prefix": "gA_1_"},  2:  {"prefix": "gA_2_"},  3:  {"prefix": "gA_3_"},
    4:  {"prefix": "gA_4_"},  5:  {"prefix": "gA_5_"},  6:  {"prefix": "gB_6_"},
    7:  {"prefix": "gB_7_"},  8:  {"prefix": "gB_8_"},  9:  {"prefix": "gB_9_"},
    10: {"prefix": "gB_10_"}, 11: {"prefix": "gC_11_"}, 12: {"prefix": "gC_12_"},
    13: {"prefix": "gC_13_"}, 14: {"prefix": "gC_14_"}, 15: {"prefix": "gC_15_"},
}

SCRIPT_DIR = Path(__file__).parent
DATASET_BASE = SCRIPT_DIR.parent
RESULTS_DIR = DATASET_BASE.parent / "Jurnal JNTETI" / "Results"
SWEEP_BASE = RESULTS_DIR / "decay_sweep_output"
WORKER = SCRIPT_DIR / "_lovo_worker.py"

YAWDD_ROOT = DATASET_BASE / "YawDD" / "Mirror" / "Mirror"

VENV_PYTHON = str(SCRIPT_DIR.parent.parent / "myenv" / "Scripts" / "python.exe")
if not Path(VENV_PYTHON).exists():
    VENV_PYTHON = sys.executable

MAX_WORKERS = 2


def parse_events(csv_path, event_type=None):
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if event_type and row.get("event_type") != event_type:
                continue
            try:
                fs = int(row["frame_start"])
                fe = int(row["frame_end"])
                events.append((fs, fe))
            except (ValueError, KeyError):
                pass
    return events


def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def get_dmd_videos():
    video_files = []
    for ds_id in sorted(DS_MAPPING.keys()):
        folder = DATASET_BASE / f"DMD Dataset {ds_id}"
        vids = list(folder.glob("*_ir_face.mp4")) or list(folder.glob("*.mp4"))
        if vids:
            video_files.append((ds_id, vids[0]))
    return video_files


def get_yawdd_talking_videos():
    videos = []
    for root, _, files in os.walk(YAWDD_ROOT):
        for f in files:
            if not f.endswith(".avi"):
                continue
            name_lower = f.lower()
            if "-talking." in name_lower and "yawning" not in name_lower:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def run_single_video(video_path, output_dir, acc_override):
    w_eye, w_mouth, w_head = WEIGHTS
    cmd = [
        VENV_PYTHON, str(WORKER),
        str(video_path), str(output_dir),
        str(w_eye), str(w_mouth), str(w_head),
        str(MAR_THRESH), str(EAR_THRESH),
    ]
    if acc_override is not None:
        cmd.append(json.dumps(acc_override))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(SCRIPT_DIR))
        return {
            "video": os.path.basename(video_path),
            "returncode": result.returncode,
            "stderr": (result.stderr or "")[-300:],
        }
    except Exception as e:
        return {"video": os.path.basename(video_path), "returncode": -1, "stderr": str(e)}


def evaluate_dmd(config_dir):
    total_tp, total_fp, total_fn = 0, 0, 0
    total_alerts = 0
    total_gt = 0

    for ds_id, info in DS_MAPPING.items():
        ds_folder = DATASET_BASE / f"DMD Dataset {ds_id}"
        yawn_gt = parse_events(ds_folder / "yawning.csv")
        blink_gt = parse_events(ds_folder / "blink.csv")
        gt_events = yawn_gt + blink_gt

        pred_csvs = list(Path(config_dir).glob(f"{info['prefix']}*_events.csv"))
        if not pred_csvs:
            continue
        pred_alerts = parse_events(pred_csvs[0], event_type="fatigue_alert")
        total_alerts += len(pred_alerts)
        total_gt += len(gt_events)

        matched_gt = set()
        matched_pred = set()
        for p_idx, (ps, pe) in enumerate(pred_alerts):
            for g_idx, (gs, ge) in enumerate(gt_events):
                if overlaps(ps, pe, gs, ge):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)

        total_tp += len(matched_gt)
        total_fp += (len(pred_alerts) - len(matched_pred))
        total_fn += (len(gt_events) - len(matched_gt))

    precision = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0
    recall = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    fpr = (total_fp / total_alerts) if total_alerts > 0 else 0
    fnr = (total_fn / total_gt) if total_gt > 0 else 0

    return {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2),
        "FPR": round(fpr * 100, 2),
        "FNR": round(fnr * 100, 2),
        "Total Alerts": total_alerts,
    }


def evaluate_yawdd(config_dir, yawdd_videos):
    total_alerts = 0
    for vp in yawdd_videos:
        stem = Path(vp).stem
        events_csv = Path(config_dir) / f"{stem}_events.csv"
        alerts = parse_events(events_csv, event_type="fatigue_alert")
        total_alerts += len(alerts)

    sessions = len(yawdd_videos)
    fp_per_session = (total_alerts / sessions) if sessions > 0 else 0
    return {
        "YawDD Alerts": total_alerts,
        "YawDD Sessions": sessions,
        "YawDD FP/Session": round(fp_per_session, 3),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SWEEP_BASE, exist_ok=True)

    dmd_videos = get_dmd_videos()
    yawdd_videos = get_yawdd_talking_videos()

    tasks = []
    for cfg in DECAY_CONFIGS:
        out_dir = SWEEP_BASE / cfg["name"] / "dmd"
        for _, vf in dmd_videos:
            tasks.append((str(vf), str(out_dir), cfg["override"]))

    print(f"[INFO] Decay sweep DMD tasks: {len(tasks)}")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_video, *t): t for t in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            status = "OK" if res["returncode"] == 0 else "FAIL"
            print(f"\rProgress: {completed}/{len(tasks)} | {status} {res['video']}", end="", flush=True)

    print(f"\n[INFO] DMD sweep done in {(time.time() - start_time) / 60:.1f} min")

    # Run YawDD for FP/session
    yawdd_tasks = []
    for cfg in DECAY_CONFIGS:
        out_dir = SWEEP_BASE / cfg["name"] / "yawdd"
        for vp in yawdd_videos:
            yawdd_tasks.append((vp, str(out_dir), cfg["override"]))

    print(f"[INFO] Decay sweep YawDD tasks: {len(yawdd_tasks)}")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_video, *t): t for t in yawdd_tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            status = "OK" if res["returncode"] == 0 else "FAIL"
            print(f"\rProgress: {completed}/{len(yawdd_tasks)} | {status} {res['video']}", end="", flush=True)

    print(f"\n[INFO] YawDD sweep done in {(time.time() - start_time) / 60:.1f} min")

    # Evaluate and write summary
    summary = []
    for cfg in DECAY_CONFIGS:
        dmd_dir = SWEEP_BASE / cfg["name"] / "dmd"
        yawdd_dir = SWEEP_BASE / cfg["name"] / "yawdd"
        dmd_metrics = evaluate_dmd(dmd_dir)
        yawdd_metrics = evaluate_yawdd(yawdd_dir, yawdd_videos)

        row = {
            "Config": cfg["name"],
            **dmd_metrics,
            **yawdd_metrics,
        }
        summary.append(row)

    out_csv = RESULTS_DIR / "exp12_decay_sensitivity.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"[DONE] Results saved to {out_csv}")
    for row in summary:
        print(f"{row['Config']:>12}: F1={row['F1']:>6}% | FP/Session={row['YawDD FP/Session']:>5}")


if __name__ == "__main__":
    main()
