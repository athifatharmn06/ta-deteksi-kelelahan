"""
exp11_weight_sensitivity.py
Experiment 11: Weight Sensitivity Sweep — PARALLELIZED
Evaluates 4 weight configurations for the drowsiness score across 15 DMD videos.
"""
import subprocess
import sys
import os
import csv
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Configuration ────────────────────────────────────────────────────────
WEIGHT_CONFIGS = [
    {"name": "Equal",        "weights": [0.33, 0.33, 0.34]}, # Equal weighting (baseline)
    {"name": "Proposed",     "weights": [0.45, 0.25, 0.30]}, # Proposed (hierarchy-based)
    {"name": "Alt1",         "weights": [0.40, 0.30, 0.30]}, # Alternative 1
    {"name": "Alt2",         "weights": [0.50, 0.20, 0.30]}, # Alternative 2
    {"name": "Alt3",         "weights": [0.35, 0.35, 0.30]}, # Alternative 3
]

# Thresholds from user's "Experiment 01" optimal values
MAR_THRESH = 0.26
EAR_THRESH = 0.16

DS_MAPPING = {
    1:  {"prefix": "gA_1_"},  2:  {"prefix": "gA_2_"},  3:  {"prefix": "gA_3_"},
    4:  {"prefix": "gA_4_"},  5:  {"prefix": "gA_5_"},  6:  {"prefix": "gB_6_"},
    7:  {"prefix": "gB_7_"},  8:  {"prefix": "gB_8_"},  9:  {"prefix": "gB_9_"},
    10: {"prefix": "gB_10_"}, 11: {"prefix": "gC_11_"}, 12: {"prefix": "gC_12_"},
    13: {"prefix": "gC_13_"}, 14: {"prefix": "gC_14_"}, 15: {"prefix": "gC_15_"},
}

def _find_python():
    # Check for venv explicitly in JNTETI structure
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    # Try c:\TA Deteksi Kelelahan\myenv
    venv_python = script_dir.parent.parent / "myenv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable

PYTHON_EXE = _find_python()

def parse_events(csv_path, event_type=None):
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if event_type and row.get("event_type") != event_type:
                    continue
                fs = int(row["frame_start"])
                fe = int(row["frame_end"])
                events.append((fs, fe))
            except (ValueError, KeyError):
                pass
    return events

def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)

def run_single_video(args):
    video_path, output_dir, weights, worker_script, cwd = args
    w_eye, w_mouth, w_head = weights
    
    cmd = [
        PYTHON_EXE, str(worker_script),
        str(video_path), str(output_dir),
        str(w_eye), str(w_mouth), str(w_head),
        str(MAR_THRESH), str(EAR_THRESH)
    ]
    
    try:
        # Increase timeout as real video processing is slower
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(cwd), env=os.environ.copy())
        return {
            "video": os.path.basename(video_path),
            "weights": weights,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"video": os.path.basename(video_path), "weights": weights, "returncode": -1, "error": str(e)}

def evaluate_weight_config(config_dir, dataset_base):
    total_tp, total_fp, total_fn = 0, 0, 0
    total_alerts = 0
    total_gt = 0
    
    for ds_id, info in DS_MAPPING.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        # Combine GT from yawning and blink (as both are fatigue signs)
        yawn_gt = parse_events(ds_folder / "yawning.csv")
        blink_gt = parse_events(ds_folder / "blink.csv")
        gt_events = yawn_gt + blink_gt
        
        # Predicted Alerts from v5_own_detection_model's events.csv
        # MDL saves events in output_dir/[prefix]_events.csv
        pred_csvs = list(Path(config_dir).glob(f"{info['prefix']}*_events.csv"))
        if not pred_csvs: continue
        
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
        
    p = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0
    r = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
    
    fnr = total_fn / total_gt if total_gt > 0 else 0
    # FPR is ill-defined without TN, we'll approximate as False Discovery Rate (FP / Total Alerts) or just ratio of false alerts
    # We will output False Discovery Rate % as FPR here per user's typical usage, or 1 - Precision.
    fpr = total_fp / total_alerts if total_alerts > 0 else 0
    
    return total_tp, total_fp, total_fn, round(p*100, 2), round(r*100, 2), round(f1*100, 2), round(fpr*100, 2), round(fnr*100, 2), total_alerts

def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    dataset_base = script_dir.parent
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    sweep_base = script_dir / "weight_sweep_output"
    worker_script = script_dir / "_weight_worker.py"
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sweep_base, exist_ok=True)
    
    video_files = []
    for ds_id in sorted(DS_MAPPING.keys()):
        folder = dataset_base / f"DMD Dataset {ds_id}"
        vids = list(folder.glob("*_ir_face.mp4")) or list(folder.glob("*.mp4"))
        if vids: video_files.append(vids[0])
        
    tasks = []
    for cfg in WEIGHT_CONFIGS:
        out_dir = sweep_base / cfg["name"]
        for vf in video_files:
            tasks.append((str(vf), str(out_dir), cfg["weights"], str(worker_script), str(script_dir)))
            
    print(f"[INFO] Started Experiment 11 Weight Sweep (Parallel): {len(tasks)} tasks...")
    print(f"[INFO] Using Python: {PYTHON_EXE}")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_video, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            status = "OK" if res["returncode"] == 0 else "FAIL"
            print(f"\rProgress: {completed}/{len(tasks)} | {status} {res['video']}", end="", flush=True)
            
            # Diagnostic: Print output of the first task
            if completed == 1:
                print(f"\n[DIAGNOSTIC] Output for {res['video']}:")
                if res.get('stdout'): print(f"STDOUT: {res['stdout'][-500:]}")
                if res.get('stderr'): print(f"STDERR: {res['stderr']}")
            
    print(f"\n[INFO] Sweep completed in {(time.time()-start_time)/60:.1f} min. Evaluating...")
    
    final_results = []
    for cfg in WEIGHT_CONFIGS:
        tp, fp, fn, prec, rec, f1, fpr, fnr, total_alerts = evaluate_weight_config(sweep_base / cfg["name"], dataset_base)
        final_results.append({
            "Config": cfg["name"],
            "Weights": f"{cfg['weights'][0]}/{cfg['weights'][1]}/{cfg['weights'][2]}",
            "F1-Score": f1,
            "False Positive Rate": fpr,
            "False Negative Rate": fnr,
            "Total Alerts Triggered": total_alerts,
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec
        })
        
    summary_path = results_dir / "exp11_weight_sensitivity.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
        writer.writeheader()
        writer.writerows(final_results)
        
    print(f"[DONE] Results saved to {summary_path}")
    for res in final_results:
        print(f"{res['Config']:>12}: F1={res['F1-Score']:>6}% | Total Alerts={res['Total Alerts Triggered']:>3} | FPR={res['False Positive Rate']:>6}% | FNR={res['False Negative Rate']:>6}%")

if __name__ == "__main__":
    main()
