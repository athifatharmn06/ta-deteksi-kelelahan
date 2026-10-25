"""
exp2_threshold_sensitivity.py
Experiment 2: Threshold Sensitivity Analysis — PARALLELIZED

Runs the full system (C5 config) across all 15 videos with multiple
MAR and EAR threshold values to measure sensitivity.

Hardware-optimized for: i5-11300H (4C/8T), 16GB RAM, RTX 3050
Strategy: 4 parallel subprocess workers, each isolated process

Usage:
    python exp2_threshold_sensitivity.py               # Run all sweeps
    python exp2_threshold_sensitivity.py --workers 2    # Limit workers
    python exp2_threshold_sensitivity.py --mar-only     # Only MAR sweep
    python exp2_threshold_sensitivity.py --ear-only     # Only EAR sweep
"""
import subprocess
import sys
import os
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# ── Auto-detect venv Python ─────────────────────────────────────────────
def _find_python():
    """Find the correct Python with mediapipe installed."""
    # Check for venv relative to project root
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    project_root = script_dir.parent.parent  # c:\TA Deteksi Kelelahan
    venv_python = project_root / "myenv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # Fallback to current Python
    return sys.executable

PYTHON_EXE = _find_python()


# ── Configuration ────────────────────────────────────────────────────────
MAR_THRESHOLDS = [0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26]
EAR_THRESHOLDS = [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.22]

DS_MAPPING = {
    1:  {"group": "gA", "prefix": "gA_1_"},
    2:  {"group": "gA", "prefix": "gA_2_"},
    3:  {"group": "gA", "prefix": "gA_3_"},
    4:  {"group": "gA", "prefix": "gA_4_"},
    5:  {"group": "gA", "prefix": "gA_5_"},
    6:  {"group": "gB", "prefix": "gB_6_"},
    7:  {"group": "gB", "prefix": "gB_7_"},
    8:  {"group": "gB", "prefix": "gB_8_"},
    9:  {"group": "gB", "prefix": "gB_9_"},
    10: {"group": "gB", "prefix": "gB_10_"},
    11: {"group": "gC", "prefix": "gC_11_"},
    12: {"group": "gC", "prefix": "gC_12_"},
    13: {"group": "gC", "prefix": "gC_13_"},
    14: {"group": "gC", "prefix": "gC_14_"},
    15: {"group": "gC", "prefix": "gC_15_"},
}

FPS = 30


# ── Worker script (written to temp file, run as subprocess) ──────────────
WORKER_SCRIPT = '''
"""Threshold sweep worker — runs a SINGLE video with specific thresholds."""
import sys, os, json

# Parse args
video_path = sys.argv[1]
output_dir = sys.argv[2]
mar_threshold = float(sys.argv[3])
ear_threshold = float(sys.argv[4])

os.makedirs(output_dir, exist_ok=True)

# Import and configure
import v5_own_detection_model as mdl
mdl.USE_OSCILLATION_GUARD = True
mdl.USE_PEAK_GATE = True
mdl.USE_PERCLOS_WARMUP = True
mdl.THRESH_MAR_YAWN = mar_threshold
mdl.THRESH_EAR_CLOSE = ear_threshold

# Process
try:
    mdl.process_video(video_path, output_dir, headless=True)
    print(json.dumps({"status": "ok", "video": os.path.basename(video_path)}))
except Exception as e:
    print(json.dumps({"status": "error", "video": os.path.basename(video_path), "error": str(e)}))
'''


def parse_events(csv_path, start_col="frame_start", end_col="frame_end"):
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fs = int(row[start_col])
                fe = int(row[end_col])
                events.append((fs, fe))
            except (ValueError, KeyError):
                pass
    return events


def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


def evaluate_config(config_dir, dataset_base):
    """Evaluate a single config directory against all GT."""
    total_tp, total_fp, total_fn = 0, 0, 0
    per_video = []

    for ds_id, info in DS_MAPPING.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        gt_csv = ds_folder / "yawning.csv"
        if not gt_csv.exists():
            continue

        gt_events = parse_events(gt_csv)
        pred_csvs = list(Path(config_dir).glob(f"{info['prefix']}*_yawning.csv"))
        if not pred_csvs:
            continue

        pred_events = parse_events(pred_csvs[0])

        matched_gt = set()
        matched_pred = set()
        for p_idx, (ps, pe) in enumerate(pred_events):
            for g_idx, (gs, ge) in enumerate(gt_events):
                if overlaps(ps, pe, gs, ge):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)

        tp = len(matched_gt)
        fp = len(pred_events) - len(matched_pred)
        fn = len(gt_events) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        per_video.append({
            "dataset": f"DS{ds_id:02d}",
            "group": info["group"],
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 2), "recall": round(r, 2), "f1": round(f1, 2),
        })

    return total_tp, total_fp, total_fn, per_video


def run_single_video(args):
    """Run one video with specific thresholds via subprocess."""
    video_path, output_dir, mar_th, ear_th, worker_script_path, cwd = args

    cmd = [
        PYTHON_EXE, str(worker_script_path),
        str(video_path), str(output_dir),
        str(mar_th), str(ear_th),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            timeout=600,  # 10 minute timeout per video
            cwd=str(cwd),
            env=env,
        )
        return {
            "video": os.path.basename(video_path),
            "mar_th": mar_th,
            "ear_th": ear_th,
            "returncode": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "video": os.path.basename(video_path),
            "mar_th": mar_th,
            "ear_th": ear_th,
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT after 600s",
        }
    except Exception as e:
        return {
            "video": os.path.basename(video_path),
            "mar_th": mar_th,
            "ear_th": ear_th,
            "returncode": -2,
            "stdout": "",
            "stderr": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Threshold Sensitivity Sweep")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--mar-only", action="store_true",
                        help="Only run MAR threshold sweep")
    parser.add_argument("--ear-only", action="store_true",
                        help="Only run EAR threshold sweep")
    args = parser.parse_args()

    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    dataset_base = script_dir.parent
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    sweep_base = script_dir / "threshold_sweep_output"
    os.makedirs(results_dir, exist_ok=True)

    # Use standalone worker script (must exist already)
    worker_script_path = script_dir / "_threshold_worker.py"
    if not worker_script_path.exists():
        print(f"[FATAL] Worker script not found: {worker_script_path}")
        return
    print(f"[INFO] Worker script: {worker_script_path}")
    print(f"[INFO] Python executable: {PYTHON_EXE}")

    # Verify worker script exists
    assert worker_script_path.exists(), f"Worker script not found: {worker_script_path}"

    # Discover all videos
    video_files = []
    for ds_id in sorted(DS_MAPPING.keys()):
        folder = dataset_base / f"DMD Dataset {ds_id}"
        if folder.exists():
            vids = list(folder.glob("*_ir_face.mp4"))
            if not vids:
                vids = list(folder.glob("*.mp4"))
            if vids:
                video_files.append(vids[0])

    print(f"[INFO] Found {len(video_files)} videos")
    print(f"[INFO] Workers: {args.workers}")
    print(f"[INFO] CPU: i5-11300H (4C/8T)")
    print(f"[INFO] RAM: 16 GB")
    print(f"[INFO] GPU: RTX 3050 (MediaPipe uses CPU)")

    # Quick smoke test: run worker on first video
    print(f"[INFO] Smoke test: running worker on first video...")
    test_out = sweep_base / "_smoke_test"
    os.makedirs(test_out, exist_ok=True)
    test_result = run_single_video((
        str(video_files[0]), str(test_out), 0.20, 0.18,
        str(worker_script_path), str(script_dir)
    ))
    if test_result["returncode"] != 0:
        print(f"[ERROR] Smoke test FAILED!")
        print(f"  stdout: {test_result['stdout']}")
        print(f"  stderr: {test_result['stderr']}")
        print(f"[ABORT] Fix the worker script issue before running full sweep.")
        return
    else:
        print(f"[OK] Smoke test passed!")

    # ── Build task list ──────────────────────────────────────────────────
    tasks = []

    if not args.ear_only:
        # MAR sweep (keep EAR at default 0.18)
        for mar_th in MAR_THRESHOLDS:
            out_dir = sweep_base / f"MAR_{mar_th:.2f}"
            os.makedirs(out_dir, exist_ok=True)
            for vf in video_files:
                tasks.append((str(vf), str(out_dir), mar_th, 0.18,
                            str(worker_script_path), str(script_dir)))

    if not args.mar_only:
        # EAR sweep (keep MAR at default 0.20)
        for ear_th in EAR_THRESHOLDS:
            out_dir = sweep_base / f"EAR_{ear_th:.2f}"
            os.makedirs(out_dir, exist_ok=True)
            for vf in video_files:
                tasks.append((str(vf), str(out_dir), 0.20, ear_th,
                            str(worker_script_path), str(script_dir)))

    total_tasks = len(tasks)
    print(f"\n{'='*70}")
    print(f"EXPERIMENT 2: THRESHOLD SENSITIVITY — PARALLEL SWEEP")
    print(f"Total tasks: {total_tasks} ({total_tasks // len(video_files)} configs × {len(video_files)} videos)")
    print(f"Estimated time: ~{total_tasks * 5 / args.workers / 60:.1f} hours with {args.workers} workers")
    print(f"{'='*70}\n")

    # ── Execute in parallel ──────────────────────────────────────────────
    completed = 0
    errors = 0
    start_time = time.time()

    # Use ProcessPoolExecutor for parallel subprocess management
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_video, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            result = future.result()
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_tasks - completed) if completed > 0 else 0

            if result["returncode"] != 0:
                errors += 1
                status = "X"
                if errors <= 3: # Print detailed error for the first few errors
                    print(f"\n[ERR] {result['video']} MAR={result['mar_th']:.2f} EAR={result['ear_th']:.2f}: {result['stderr'][:200]}")
            else:
                status = "OK"

            # Progress bar
            pct = completed / total_tasks * 100
            bar_len = 30
            filled = int(bar_len * completed / total_tasks)
            bar = "#" * filled + "." * (bar_len - filled)

            print(f"\r[{bar}] {pct:5.1f}% | {completed}/{total_tasks} | "
                  f"ETA: {eta/60:.0f}min | Err: {errors} | "
                  f"{status} MAR={result['mar_th']:.2f} EAR={result['ear_th']:.2f} {result['video'][:30]}",
                  end="", flush=True)

    print(f"\n\n[INFO] All tasks completed in {(time.time()-start_time)/60:.1f} minutes")
    print(f"[INFO] Errors: {errors}/{total_tasks}")

    # ── Evaluate all sweep configs ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("EVALUATION PHASE")
    print(f"{'='*70}")

    all_results = []

    # MAR sweep results
    if not args.ear_only:
        print(f"\n--- MAR Threshold Sweep (EAR fixed at 0.18) ---")
        print(f"{'MAR':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 50)

        for mar_th in MAR_THRESHOLDS:
            config_dir = sweep_base / f"MAR_{mar_th:.2f}"
            tp, fp, fn, per_video = evaluate_config(config_dir, dataset_base)
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0

            marker = " <-- default" if mar_th == 0.20 else ""
            print(f"{mar_th:>6.2f} {tp:>4} {fp:>4} {fn:>4} {p:>9.1f}% {r:>7.1f}% {f1:>7.1f}%{marker}")

            all_results.append({
                "sweep_type": "MAR",
                "threshold": mar_th,
                "fixed_param": "EAR=0.18",
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 2),
                "recall": round(r, 2),
                "f1": round(f1, 2),
                "per_video": per_video,
            })

    # EAR sweep results
    if not args.mar_only:
        print(f"\n--- EAR Threshold Sweep (MAR fixed at 0.20) ---")
        print(f"{'EAR':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 50)

        for ear_th in EAR_THRESHOLDS:
            config_dir = sweep_base / f"EAR_{ear_th:.2f}"
            tp, fp, fn, per_video = evaluate_config(config_dir, dataset_base)
            p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0

            marker = " <-- default" if ear_th == 0.18 else ""
            print(f"{ear_th:>6.2f} {tp:>4} {fp:>4} {fn:>4} {p:>9.1f}% {r:>7.1f}% {f1:>7.1f}%{marker}")

            all_results.append({
                "sweep_type": "EAR",
                "threshold": ear_th,
                "fixed_param": "MAR=0.20",
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 2),
                "recall": round(r, 2),
                "f1": round(f1, 2),
                "per_video": per_video,
            })

    # ── Per-group analysis for MAR sweep ─────────────────────────────────
    if not args.ear_only:
        print(f"\n--- MAR Sweep: F1 by Group ---")
        print(f"{'MAR':>6}", end="")
        for grp in ["gA", "gB", "gC"]:
            print(f" {'F1_'+grp:>8}", end="")
        print(f" {'F1_ALL':>8}")
        print("-" * 40)

        for r in all_results:
            if r["sweep_type"] != "MAR":
                continue
            print(f"{r['threshold']:>6.2f}", end="")
            for grp in ["gA", "gB", "gC"]:
                grp_vids = [v for v in r["per_video"] if v["group"] == grp]
                if grp_vids:
                    f1s = [v["f1"] for v in grp_vids]
                    print(f" {np.mean(f1s):>7.1f}%", end="")
                else:
                    print(f" {'N/A':>8}", end="")
            print(f" {r['f1']:>7.1f}%")

    # ── Save results ─────────────────────────────────────────────────────
    # Summary CSV
    summary_csv = results_dir / "exp2_threshold_sensitivity.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["sweep_type", "threshold", "fixed_param", "tp", "fp", "fn",
                  "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n[SAVED] {summary_csv}")

    # Per-video detail CSV
    detail_csv = results_dir / "exp2_threshold_per_video.csv"
    detail_rows = []
    for r in all_results:
        for v in r["per_video"]:
            detail_rows.append({
                "sweep_type": r["sweep_type"],
                "threshold": r["threshold"],
                **v,
            })
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["sweep_type", "threshold", "dataset", "group",
                  "tp", "fp", "fn", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(detail_rows)
    print(f"[SAVED] {detail_csv}")

    # JSON summary
    json_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "per_video"}
        json_results.append(entry)

    stats_json = results_dir / "exp2_threshold_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"[SAVED] {stats_json}")

    print(f"\n[DONE] Experiment 2 complete!")
    print(f"Total runtime: {(time.time()-start_time)/60:.1f} minutes")


if __name__ == "__main__":
    main()
