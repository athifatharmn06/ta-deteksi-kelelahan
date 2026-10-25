"""
ablation_orchestrator.py — Run 4 ablation configs on DMD + YawDD Talking.
Outputs final table to JSON and console.

WHAT IT DOES:
  1. DMD: Process 15 videos x 4 configs, evaluate against GT=38 (without hand)
  2. YawDD Talking: Process ~100 Talking videos x 4 configs, count FP rate
"""
import subprocess
import json
import csv
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ============================================================
# PATHS
# ============================================================
VENV_PYTHON = r"C:\TA Deteksi Kelelahan\myenv\Scripts\python.exe"
WORKER      = r"C:\TA Deteksi Kelelahan\DataSet\dataset code\ablation_worker.py"
OUT_BASE    = r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\ablation_runs"
DATASET_DIR = r"C:\TA Deteksi Kelelahan\DataSet"

# DMD video list
DMD_LIST    = r"C:\TA Deteksi Kelelahan\DataSet\dataset code\dmd_video_list.txt"

# YawDD Talking root
YAWDD_ROOT  = r"C:\TA Deteksi Kelelahan\DataSet\YawDD\Mirror\Mirror"

# Configs to test
CONFIGS = ['C1', 'C2', 'C3', 'C4']

# Max parallel workers (limited by GPU/MediaPipe memory)
MAX_WORKERS = 2

# ============================================================
# DMD Ground Truth (without-hand yawning events only)
# ============================================================
# From exp7_fn_classified.csv: without-hand FN per DS
# From exp3_per_video_detail.csv: gt_without_hand per DS
DMD_GT_WITHOUT_HAND = {
    'DS01': {'gt_wo': 3, 'fn_ids': []},  # will load from CSV
    'DS02': {'gt_wo': 4, 'fn_ids': []},
    'DS03': {'gt_wo': 2, 'fn_ids': []},
    'DS04': {'gt_wo': 2, 'fn_ids': []},
    'DS05': {'gt_wo': 2, 'fn_ids': []},
    'DS06': {'gt_wo': 2, 'fn_ids': []},
    'DS07': {'gt_wo': 4, 'fn_ids': []},
    'DS08': {'gt_wo': 2, 'fn_ids': []},
    'DS09': {'gt_wo': 2, 'fn_ids': []},
    'DS10': {'gt_wo': 4, 'fn_ids': []},
    'DS11': {'gt_wo': 2, 'fn_ids': []},
    'DS12': {'gt_wo': 2, 'fn_ids': []},
    'DS13': {'gt_wo': 2, 'fn_ids': []},
    'DS14': {'gt_wo': 3, 'fn_ids': []},
    'DS15': {'gt_wo': 2, 'fn_ids': []},
}

def load_dmd_gt():
    """Load GT yawning events (without hand ONLY) from annotation CSVs."""
    gt_per_ds = {}
    for i in range(1, 16):
        ds_dir = Path(DATASET_DIR) / f"DMD Dataset {i}"
        gt_csv = ds_dir / "yawning.csv"
        events = []
        if gt_csv.exists():
            with open(gt_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    action = row.get('action_type', '')
                    if 'without hand' in action.lower():
                        events.append({
                            'frame_start': int(row['frame_start']),
                            'frame_end':   int(row['frame_end']),
                        })
        gt_per_ds[f"DS{i:02d}"] = events
    return gt_per_ds

def load_dmd_gt_simple():
    """Load GT from exp3_per_video_detail.csv for gt_without_hand counts."""
    csv_path = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\exp3_per_video_detail.csv")
    gt = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row['dataset']] = int(row['gt_without_hand'])
    return gt


def run_worker(video_path, config, output_dir):
    """Run ablation_worker.py via subprocess, return parsed result from file."""
    video_stem = Path(video_path).stem
    result_file = Path(output_dir) / config / video_stem / "_ablation_result.json"
    
    # Skip if result already exists (Resume Mode)
    if result_file.exists():
        try:
            with open(result_file) as f:
                cached_res = json.load(f)
                if 'error' not in cached_res:
                    return cached_res
        except:
            pass # Re-run if file is corrupted

    cmd = [VENV_PYTHON, WORKER, video_path, config, output_dir]
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            cwd=str(Path(WORKER).parent),
            env=env
        )
        
        # Read result from file (worker writes here to avoid stdout pollution)
        if result_file.exists():
            with open(result_file) as f:
                return json.load(f)
        else:
            return {
                "error": f"No result file. Exit code: {result.returncode}",
                "stderr_tail": (result.stderr or "")[-300:],
                "video": os.path.basename(video_path),
                "config": config,
            }
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (600s)", "video": os.path.basename(video_path), "config": config}
    except Exception as e:
        return {"error": str(e), "video": os.path.basename(video_path), "config": config}



def evaluate_dmd(detected_events, gt_events):
    """
    Compare detected yawn events vs GT yawn events (without-hand only).
    Uses frame overlap: detected TP if any GT event overlaps by >= 1 frame.
    Returns: tp, fp (detection-level), fn (gt-level)
    """
    gt_matched = set()
    det_matched = set()
    
    for di, det in enumerate(detected_events):
        for gi, gt in enumerate(gt_events):
            # Check overlap
            overlap_start = max(det['frame_start'], gt['frame_start'])
            overlap_end   = min(det['frame_end'],   gt['frame_end'])
            if overlap_end >= overlap_start:  # at least 1 frame overlap
                gt_matched.add(gi)
                det_matched.add(di)
    
    tp = len(gt_matched)
    fp = len(detected_events) - len(det_matched)
    fn = len(gt_events) - len(gt_matched)
    return tp, fp, fn


def get_yawdd_talking_videos():
    """Get all YawDD Mirror Talking videos (excluding TalkingYawning)."""
    videos = []
    for root, dirs, files in os.walk(YAWDD_ROOT):
        for f in files:
            if not f.endswith('.avi'):
                continue
            name_lower = f.lower()
            # Must contain "talking" but NOT "talkingyawning" or "yawning"
            if '-talking.' in name_lower and 'yawning' not in name_lower:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def get_dmd_videos():
    """Load DMD video paths from list file."""
    import re
    videos = []
    with open(DMD_LIST, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                ds_name, path = line.split('|', 1)
                # Extract number from "DMD Dataset X"
                m = re.search(r'(\d+)', ds_name)
                if m:
                    ds_id = f"DS{int(m.group(1)):02d}"
                    videos.append((ds_id, path))
    return videos


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    
    # ─── PHASE 1: DMD ────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: DMD EVALUATION (15 videos x 4 configs)")
    print("=" * 70)
    
    dmd_videos = get_dmd_videos()
    print(f"Found {len(dmd_videos)} DMD videos")
    
    # Load GT
    gt_per_ds = load_dmd_gt()
    gt_counts = load_dmd_gt_simple()
    
    # Check GT loaded
    total_gt = sum(len(v) for v in gt_per_ds.values())
    print(f"Total GT events (without hand): {total_gt}")
    
    if total_gt == 0:
        print("[WARN] GT events not loaded from annotation CSVs.")
        print("       Will use detection count only. Manual GT comparison needed.")
    
    dmd_results = {}  # config -> {ds_id -> result}
    
    # Submit all DMD jobs
    jobs = []
    for config in CONFIGS:
        dmd_results[config] = {}
        for ds_id, video_path in dmd_videos:
            jobs.append((ds_id, video_path, config))
    
    print(f"\nSubmitting {len(jobs)} DMD jobs (max {MAX_WORKERS} parallel)...")
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ds_id, video_path, config in jobs:
            future = executor.submit(run_worker, video_path, config, OUT_BASE)
            futures[future] = (ds_id, config)
        
        done = 0
        for future in as_completed(futures):
            ds_id, config = futures[future]
            result = future.result()
            dmd_results[config][ds_id] = result
            done += 1
            yc = result.get('yawn_count', '?')
            err = result.get('error', '')
            status = f"ERR: {err[:50]}" if err else f"yawns={yc}"
            print(f"  [{done}/{len(jobs)}] {config} {ds_id}: {status}")
    
    dmd_time = time.time() - t0
    print(f"\nDMD completed in {dmd_time:.0f}s")
    
    # ─── EVALUATE DMD ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DMD EVALUATION TABLE (GT=38, Without Hand Only)")
    print("=" * 70)
    
    dmd_table = {}
    for config in CONFIGS:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt_count = 0
        
        for ds_id, video_path in dmd_videos:
            r = dmd_results[config].get(ds_id, {})
            if 'error' in r:
                print(f"  [SKIP] {config} {ds_id}: {r['error'][:60]}")
                continue
            
            det_events = r.get('yawn_events', [])
            gt_events  = gt_per_ds.get(ds_id, [])
            gt_wo      = gt_counts.get(ds_id, 0)
            
            if gt_events:
                tp, fp, fn = evaluate_dmd(det_events, gt_events)
            else:
                # Fallback: use count-based comparison
                tp = min(len(det_events), gt_wo)
                fp = max(0, len(det_events) - gt_wo)
                fn = max(0, gt_wo - len(det_events))
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt_count += gt_wo
        
        if (total_tp + total_fp) > 0:
            precision = total_tp / (total_tp + total_fp) * 100
        else:
            precision = 0
        
        if total_gt_count > 0:
            recall = total_tp / total_gt_count * 100
        else:
            recall = 0
        
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        dmd_table[config] = {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'gt': total_gt_count,
            'precision': round(precision, 1),
            'recall': round(recall, 1),
            'f1': round(f1, 1),
        }
        print(f"  {config}: TP={total_tp} FP={total_fp} FN={total_fn} "
              f"P={precision:.1f}% R={recall:.1f}% F1={f1:.1f}%")
    
    # ─── PHASE 2: YawDD TALKING ──────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: YawDD TALKING EVALUATION (4 configs)")
    print("=" * 70)
    
    talking_videos = get_yawdd_talking_videos()
    print(f"Found {len(talking_videos)} YawDD Talking videos")
    
    if len(talking_videos) == 0:
        print("[ERROR] No Talking videos found!")
        # Try alternative pattern
        talking_videos_alt = []
        for root, dirs, files in os.walk(YAWDD_ROOT):
            for f in files:
                if f.endswith('.avi') and 'talking' in f.lower() and 'yawning' not in f.lower():
                    talking_videos_alt.append(os.path.join(root, f))
        print(f"  Alternative search found: {len(talking_videos_alt)} videos")
        talking_videos = sorted(talking_videos_alt)
    
    yawdd_results = {}  # config -> [results]
    
    # Submit all YawDD jobs
    jobs_y = []
    for config in CONFIGS:
        yawdd_results[config] = []
        for vpath in talking_videos:
            jobs_y.append((vpath, config))
    
    print(f"\nSubmitting {len(jobs_y)} YawDD jobs (max {MAX_WORKERS} parallel)...")
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for vpath, config in jobs_y:
            future = executor.submit(run_worker, vpath, config, OUT_BASE)
            futures[future] = (vpath, config)
        
        done = 0
        for future in as_completed(futures):
            vpath, config = futures[future]
            result = future.result()
            yawdd_results[config].append(result)
            done += 1
            if done % 20 == 0 or done == len(jobs_y):
                print(f"  [{done}/{len(jobs_y)}] completed...")
    
    yawdd_time = time.time() - t0
    print(f"\nYawDD completed in {yawdd_time:.0f}s")
    
    # ─── EVALUATE YAWDD ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("YawDD TALKING FP RATE")
    print("=" * 70)
    
    yawdd_table = {}
    for config in CONFIGS:
        results = yawdd_results[config]
        total   = len([r for r in results if 'error' not in r])
        errors  = len([r for r in results if 'error' in r])
        triggered = len([r for r in results if r.get('yawn_count', 0) > 0 and 'error' not in r])
        
        fp_rate = triggered / total * 100 if total > 0 else 0
        
        yawdd_table[config] = {
            'total': total,
            'triggered': triggered,
            'errors': errors,
            'fp_rate': round(fp_rate, 1),
        }
        print(f"  {config}: {triggered}/{total} videos triggered ({fp_rate:.1f}% FP rate)"
              f"{'  ['+str(errors)+' errors]' if errors else ''}")
    
    # ─── FINAL COMBINED TABLE ────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL ABLATION TABLE")
    print("=" * 70)
    
    header = (f"{'Config':<25} | {'DMD TP':>6} {'DMD FP':>6} {'DMD FN':>6} | "
              f"{'Prec':>7} {'Recall':>7} {'F1':>7} | {'YawDD FP Rate':>14}")
    print(header)
    print("-" * len(header))
    
    for config in CONFIGS:
        d = dmd_table.get(config, {})
        y = yawdd_table.get(config, {})
        print(f"{config:<25} | {d.get('tp','?'):>6} {d.get('fp','?'):>6} {d.get('fn','?'):>6} | "
              f"{d.get('precision','?'):>6}% {d.get('recall','?'):>6}% {d.get('f1','?'):>6}% | "
              f"{y.get('triggered','?')}/{y.get('total','?')} ({y.get('fp_rate','?')}%)")
    
    # ─── SAVE RESULTS ────────────────────────────────────────
    final = {
        'dmd': dmd_table,
        'yawdd_talking': yawdd_table,
        'metadata': {
            'dmd_videos': len(dmd_videos),
            'yawdd_talking_videos': len(talking_videos),
            'dmd_time_seconds': round(dmd_time),
            'yawdd_time_seconds': round(yawdd_time),
            'configs': CONFIGS,
        }
    }
    
    out_json = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\ablation_final_table.json")
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {dmd_time + yawdd_time:.0f}s "
          f"({(dmd_time + yawdd_time)/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
