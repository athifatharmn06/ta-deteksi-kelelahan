"""
exp_level_validation.py
Experiment: Validasi Threshold Level Kelelahan 30/50/70
dengan Leave-One-Video-Out (LOVO) pada 15 video DMD.

Fase 1: Score Generation — jalankan _lovo_worker.py per video per weight config
Fase 2: LOVO Evaluation — sweep threshold + weight + critical override
Fase 3: Simpan hasil CSV

Usage:
    python exp_level_validation.py               # full run
    python exp_level_validation.py --skip-gen    # skip fase 1 (pakai cache)
    python exp_level_validation.py --workers 2   # batasi paralel worker
"""

import subprocess, sys, os, json, csv, time, argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATASET_BASE = SCRIPT_DIR.parent
GT_DIR       = DATASET_BASE.parent / "Jurnal JNTETI" / "Results" / "level_validation" / "gt"
SCORES_DIR   = DATASET_BASE.parent / "Jurnal JNTETI" / "Results" / "level_validation" / "scores"
SCORES_YAWDD_DIR = DATASET_BASE.parent / "Jurnal JNTETI" / "Results" / "level_validation" / "scores_yawdd"
RESULTS_DIR  = DATASET_BASE.parent / "Jurnal JNTETI" / "Results" / "level_validation"

VENV_PYTHON  = str(SCRIPT_DIR.parent.parent / "myenv" / "Scripts" / "python.exe")
if not Path(VENV_PYTHON).exists():
    VENV_PYTHON = sys.executable

WORKER = str(SCRIPT_DIR / "_lovo_worker.py")

DS_MAPPING = {i: f"DMD Dataset {i}" for i in range(1, 16)}

# Mapping nama weight config ke folder alternatif (dari run sebelumnya di weight_sweep_output)
# Cek nama folder: Equal, Proposed, Alt1, Alt2, Alt3, Oral-Heavy, Ocular-Extra
WEIGHT_FOLDER_ALIASES = {
    "W1_Equal"   : ["W1_Equal", "Equal"],
    "W2_EyeHeavy": ["W2_EyeHeavy", "Ocular-Extra", "Alt2"],
    "W3_Proposed": ["W3_Proposed", "Proposed"],
    "W4_MaxEye"  : ["W4_MaxEye", "Alt2"],           # 0.50/0.25/0.25 paling dekat
    "W5_MouthUp" : ["W5_MouthUp", "Oral-Heavy", "Alt1"],
}

# Direktori pencarian perclos CSV (HANYA dari Fase 1 baru — jangan fallback ke cache lama)
SCORE_SEARCH_DIRS = [
    SCORES_DIR,   # satu-satunya sumber valid
]

# ── Konfigurasi Eksperimen ─────────────────────────────────────────────────────
WEIGHT_CONFIGS = [
    {"name": "W1_Equal",    "weights": (0.33, 0.33, 0.34)},
    {"name": "W2_EyeHeavy", "weights": (0.50, 0.25, 0.25)},
    {"name": "W3_Proposed", "weights": (0.45, 0.25, 0.30)},
    {"name": "W4_MaxEye",   "weights": (0.60, 0.20, 0.20)},
    {"name": "W5_MouthUp",  "weights": (0.40, 0.30, 0.30)},
]

SELECTED_WEIGHTS = {"W3_Proposed"}

ACC_OVERRIDE_NAME = "LowEventGain"
ACC_OVERRIDE = {
    "params": {
        "mouth_event_gain": 18.0,
        "head_event_gain": 16.0,
        "microsleep_event_gain": 30.0,
    }
}

ENABLE_YAWDD = True
YAWDD_ROOT = DATASET_BASE / "YawDD" / "Mirror" / "Mirror"

THRESHOLD_CONFIGS = [
    {"name": "T1_25_45_65", "thresholds": (25, 45, 65)},
    {"name": "T2_30_50_70", "thresholds": (30, 50, 70)},   # sistem saat ini
    {"name": "T3_35_55_75", "thresholds": (35, 55, 75)},
    {"name": "T4_20_40_60", "thresholds": (20, 40, 60)},
    {"name": "T5_30_60_80", "thresholds": (30, 60, 80)},
]

# Fixed sensor thresholds (dari exp2 optimal)
MAR_THRESH  = 0.20
EAR_THRESH  = 0.18
SEGMENT_SEC = 10
FPS         = 30

LABEL_NAMES = {0: "Aman", 1: "Hati-hati", 2: "Waspada", 3: "Bahaya"}


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: SCORE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_path(ds_id):
    folder = DATASET_BASE / DS_MAPPING[ds_id]
    vids = list(folder.glob("*_ir_face.mp4")) or list(folder.glob("*.mp4"))
    return str(vids[0]) if vids else None


def run_worker_task(args):
    video_path, output_dir, w_eye, w_mouth, w_head, ds_id, weight_name, acc_override_json = args
    os.makedirs(output_dir, exist_ok=True)

    # Cek cache: jika _perclos.csv sudah ada, skip
    video_stem = Path(video_path).stem
    perclos_csv = Path(output_dir) / f"{video_stem}_perclos.csv"
    if perclos_csv.exists():
        return {"status": "cached", "video": os.path.basename(video_path),
                "ds_id": ds_id, "weight_name": weight_name}

    cmd = [VENV_PYTHON, WORKER, video_path, output_dir,
           str(w_eye), str(w_mouth), str(w_head),
           str(MAR_THRESH), str(EAR_THRESH)]
    if acc_override_json:
        cmd.append(acc_override_json)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=900, cwd=str(SCRIPT_DIR), env=env)
        return {
            "status"     : "ok" if result.returncode == 0 else "error",
            "video"      : os.path.basename(video_path),
            "ds_id"      : ds_id,
            "weight_name": weight_name,
            "returncode" : result.returncode,
            "stderr"     : result.stderr[-300:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "video": os.path.basename(video_path),
                "ds_id": ds_id, "weight_name": weight_name}
    except Exception as e:
        return {"status": "error", "video": os.path.basename(video_path),
                "ds_id": ds_id, "weight_name": weight_name, "stderr": str(e)}


def phase1_generate_scores(max_workers=4):
    """Jalankan _lovo_worker untuk semua video × weight config."""
    print(f"\n{'='*65}")
    print("FASE 1: SCORE GENERATION")
    print(f"  {len(WEIGHT_CONFIGS)} weight config × 15 video = "
          f"{len(WEIGHT_CONFIGS)*15} tasks")
    print(f"  Workers: {max_workers}")
    print(f"{'='*65}")

    tasks = []
    active_weights = [w for w in WEIGHT_CONFIGS if w["name"] in SELECTED_WEIGHTS]
    acc_override_json = json.dumps(ACC_OVERRIDE) if ACC_OVERRIDE else None
    for wcfg in active_weights:
        w_eye, w_mouth, w_head = wcfg["weights"]
        out_base = SCORES_DIR / wcfg["name"]
        for ds_id in range(1, 16):
            vp = get_video_path(ds_id)
            if vp:
                tasks.append((vp, str(out_base), w_eye, w_mouth, w_head,
                               ds_id, wcfg["name"], acc_override_json))

    completed, errors, cached = 0, 0, 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_worker_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res["status"] == "cached":
                cached += 1
            elif res["status"] != "ok":
                errors += 1
                print(f"\n  [ERR] {res.get('weight_name')} DS{res.get('ds_id',0):02d}: "
                      f"{res.get('stderr','')[:120]}")

            pct = completed / len(tasks) * 100
            bar = "#" * int(pct / 3.5) + "." * (28 - int(pct / 3.5))
            print(f"\r  [{bar}] {pct:5.1f}% | {completed}/{len(tasks)} "
                  f"| err={errors} cache={cached} "
                  f"| {res.get('weight_name','?')} DS{res.get('ds_id',0):02d} "
                  f"{res['status']}", end="", flush=True)

    print(f"\n\n  Selesai dalam {(time.time()-t0)/60:.1f} menit. "
          f"Errors: {errors}/{len(tasks)}")
    return errors


def get_yawdd_talking_videos():
    videos = []
    for root, _, files in os.walk(YAWDD_ROOT):
        for f in files:
            if not f.endswith('.avi'):
                continue
            name_lower = f.lower()
            if '-talking.' in name_lower and 'yawning' not in name_lower:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def phase1_generate_scores_yawdd(max_workers=4):
    if not ENABLE_YAWDD:
        return 0

    videos = get_yawdd_talking_videos()
    if not videos:
        print("[WARN] YawDD talking videos not found.")
        return 0

    print(f"\n{'='*65}")
    print("FASE 1B: SCORE GENERATION (YawDD Talking)")
    print(f"  Videos: {len(videos)}")
    print(f"  Workers: {max_workers}")
    print(f"{'='*65}")

    tasks = []
    active_weights = [w for w in WEIGHT_CONFIGS if w["name"] in SELECTED_WEIGHTS]
    acc_override_json = json.dumps(ACC_OVERRIDE) if ACC_OVERRIDE else None
    for wcfg in active_weights:
        w_eye, w_mouth, w_head = wcfg["weights"]
        out_base = SCORES_YAWDD_DIR / wcfg["name"]
        for vp in videos:
            tasks.append((vp, str(out_base), w_eye, w_mouth, w_head,
                           0, wcfg["name"], acc_override_json))

    completed, errors, cached = 0, 0, 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_worker_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if res["status"] == "cached":
                cached += 1
            elif res["status"] != "ok":
                errors += 1

            pct = completed / len(tasks) * 100
            bar = "#" * int(pct / 3.5) + "." * (28 - int(pct / 3.5))
            print(f"\r  [{bar}] {pct:5.1f}% | {completed}/{len(tasks)} "
                  f"| err={errors} cache={cached}", end="", flush=True)

    print(f"\n\n  Selesai dalam {(time.time()-t0)/60:.1f} menit. "
          f"Errors: {errors}/{len(tasks)}")
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAS: LOAD GT LABELS & PREDICTED SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt_labels(ds_id):
    """Load GT label dari JSON yang dibuat gt_label_builder."""
    gt_path = GT_DIR / f"gt_labels_ds{ds_id:02d}.json"
    if not gt_path.exists():
        return None
    with open(gt_path, encoding='utf-8') as f:
        data = json.load(f)
    return data["segments"]   # list of dict {seg_idx, label, ...}


def load_predicted_scores(ds_id, weight_name):
    """
    Load _perclos.csv dan agregasi score per segmen SEGMENT_SEC detik.
    Cari CSV dari beberapa direktori (prioritas: SCORES_DIR → weight_sweep_output → fallback alias).
    Returns: list of float (mean score per segmen), atau None jika tidak ditemukan.
    """
    import csv as csv_mod

    ds_folder = DATASET_BASE / DS_MAPPING[ds_id]
    vids = list(ds_folder.glob("*_ir_face.mp4")) or list(ds_folder.glob("*.mp4"))
    if not vids:
        return None
    video_stem = Path(vids[0]).stem

    # Kumpulkan semua kandidat folder yang mungkin mengandung file ini
    candidate_folders = []
    aliases = WEIGHT_FOLDER_ALIASES.get(weight_name, [weight_name])
    for base_dir in SCORE_SEARCH_DIRS:
        for alias in aliases:
            candidate_folders.append(base_dir / alias)

    # Cari file perclos CSV di semua kandidat
    perclos_csv = None
    for folder in candidate_folders:
        p = folder / f"{video_stem}_perclos.csv"
        if p.exists():
            perclos_csv = p
            break

    if perclos_csv is None:
        return None

    rows = []
    with open(perclos_csv, encoding='utf-8') as f:
        for row in csv_mod.DictReader(f):
            try:
                rows.append({
                    "second": int(row["second"]),
                    "score" : float(row["score"]),
                })
            except (ValueError, KeyError):
                pass

    if not rows:
        return None

    # Agregasi per segmen SEGMENT_SEC detik
    seg_scores = []
    seg_idx = 0
    while True:
        sec_start = seg_idx * SEGMENT_SEC
        sec_end   = sec_start + SEGMENT_SEC - 1
        seg_rows  = [r["score"] for r in rows
                     if sec_start <= r["second"] <= sec_end]
        if not seg_rows:
            break
        seg_scores.append(float(np.mean(seg_rows)))
        seg_idx += 1

    return seg_scores


def load_predicted_scores_yawdd(video_path, weight_name):
    import csv as csv_mod

    video_stem = Path(video_path).stem
    candidate_folders = []
    aliases = WEIGHT_FOLDER_ALIASES.get(weight_name, [weight_name])
    for base_dir in [SCORES_YAWDD_DIR]:
        for alias in aliases:
            candidate_folders.append(base_dir / alias)

    perclos_csv = None
    for folder in candidate_folders:
        p = folder / f"{video_stem}_perclos.csv"
        if p.exists():
            perclos_csv = p
            break

    if perclos_csv is None:
        return None

    rows = []
    with open(perclos_csv, encoding='utf-8') as f:
        for row in csv_mod.DictReader(f):
            try:
                rows.append({
                    "second": int(row["second"]),
                    "score" : float(row["score"]),
                })
            except (ValueError, KeyError):
                pass

    if not rows:
        return None

    seg_scores = []
    seg_idx = 0
    while True:
        sec_start = seg_idx * SEGMENT_SEC
        sec_end   = sec_start + SEGMENT_SEC - 1
        seg_rows  = [r["score"] for r in rows
                     if sec_start <= r["second"] <= sec_end]
        if not seg_rows:
            break
        seg_scores.append(float(np.mean(seg_rows)))
        seg_idx += 1

    return seg_scores


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: LOVO EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_score(score, thresholds, critical_override=False,
                   has_microsleep=False, perclos=None, eye_closed_long=False):
    """
    Klasifikasikan score menjadi label 0-3.
    thresholds = (t_hati_hati, t_waspada, t_bahaya)
    """
    t1, t2, t3 = thresholds

    if critical_override:
        if has_microsleep:
            return 3  # Bahaya langsung
        if perclos is not None and perclos > 20.0:
            return max(2, (3 if score >= t3 else 2))
        if eye_closed_long:
            return max(2, (3 if score >= t3 else 2))

    if score >= t3:
        return 3
    elif score >= t2:
        return 2
    elif score >= t1:
        return 1
    else:
        return 0


def compute_metrics(gt_labels, pred_labels, n_classes=4):
    """Hitung accuracy, macro F1, per-class precision/recall, confusion matrix."""
    from collections import Counter

    n = len(gt_labels)
    if n == 0:
        return None

    # Confusion matrix
    cm = [[0]*n_classes for _ in range(n_classes)]
    for gt, pred in zip(gt_labels, pred_labels):
        if 0 <= gt < n_classes and 0 <= pred < n_classes:
            cm[gt][pred] += 1

    # Accuracy
    correct = sum(cm[c][c] for c in range(n_classes))
    accuracy = correct / n * 100.0

    # Per-class P / R / F1
    per_class = {}
    for c in range(n_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(n_classes)) - tp
        fn = sum(cm[c][cc] for cc in range(n_classes)) - tp
        tn = n - tp - fp - fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        far  = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm Rate

        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "false_alarm_rate": far}

    # Macro F1 (hanya kelas yang ada di GT)
    classes_in_gt = [c for c in range(n_classes)
                     if any(cm[c][cc] > 0 for cc in range(n_classes))]
    macro_f1 = (np.mean([per_class[c]["f1"] for c in classes_in_gt])
                if classes_in_gt else 0.0)

    return {
        "accuracy"     : round(accuracy, 2),
        "macro_f1"     : round(macro_f1 * 100, 2),
        "recall_bahaya": round(per_class[3]["recall"] * 100, 2),
        "false_alarm_rate": round(per_class[0]["false_alarm_rate"] * 100, 2),
        "miss_bahaya"  : round((1 - per_class[3]["recall"]) * 100, 2),
        "confusion_matrix": cm,
        "per_class"    : {str(c): {k: round(v*100, 2) if isinstance(v, float) else v
                                   for k, v in pc.items() if k != 'tp fp fn tn'.split()}
                          for c, pc in per_class.items()},
    }


def phase2_lovo_evaluation():
    """Jalankan evaluasi LOVO untuk semua kombinasi."""
    print(f"\n{'='*65}")
    print("FASE 2: LOVO EVALUATION")
    print(f"  {len(WEIGHT_CONFIGS)} weight × {len(THRESHOLD_CONFIGS)} threshold "
          f"× 2 override × 15 fold")
    print(f"{'='*65}\n")

    all_results = []
    n_combos = len(WEIGHT_CONFIGS) * len(THRESHOLD_CONFIGS) * 2

    active_weights = [w for w in WEIGHT_CONFIGS if w["name"] in SELECTED_WEIGHTS]

    yawdd_scores = {}
    yawdd_videos = []
    if ENABLE_YAWDD:
        yawdd_videos = get_yawdd_talking_videos()
        for wcfg in active_weights:
            scores = []
            for vp in yawdd_videos:
                seg_scores = load_predicted_scores_yawdd(vp, wcfg["name"])
                if seg_scores is not None:
                    scores.append(seg_scores)
            yawdd_scores[wcfg["name"]] = scores

    combo_idx = 0
    for wcfg in active_weights:
        for tcfg in THRESHOLD_CONFIGS:
            for use_override in [False, True]:
                combo_idx += 1
                override_name = "B_Override" if use_override else "A_NoOverride"

                fold_metrics = []
                fold_gt_all  = []
                fold_pred_all= []

                for ds_id in range(1, 16):
                    # Load GT
                    gt_segs = load_gt_labels(ds_id)
                    if gt_segs is None:
                        continue

                    # Load predicted scores
                    pred_scores = load_predicted_scores(ds_id, wcfg["name"])
                    if pred_scores is None:
                        continue

                    # Align: pakai panjang terpendek
                    n = min(len(gt_segs), len(pred_scores))
                    if n == 0:
                        continue

                    gt_labels   = [gt_segs[i]["label"] for i in range(n)]
                    pred_labels = [classify_score(pred_scores[i],
                                                  tcfg["thresholds"],
                                                  critical_override=use_override)
                                   for i in range(n)]

                    fold_gt_all.extend(gt_labels)
                    fold_pred_all.extend(pred_labels)

                    m = compute_metrics(gt_labels, pred_labels)
                    if m:
                        fold_metrics.append(m)

                # Aggregate across folds (macro average)
                if not fold_metrics:
                    continue

                agg_metrics = compute_metrics(fold_gt_all, fold_pred_all)
                if not agg_metrics:
                    continue

                row = {
                    "weight_config"     : wcfg["name"],
                    "w_eye"             : wcfg["weights"][0],
                    "w_mouth"           : wcfg["weights"][1],
                    "w_head"            : wcfg["weights"][2],
                    "threshold_config"  : tcfg["name"],
                    "t_hati_hati"       : tcfg["thresholds"][0],
                    "t_waspada"         : tcfg["thresholds"][1],
                    "t_bahaya"          : tcfg["thresholds"][2],
                    "critical_override" : use_override,
                    "override_mode"     : override_name,
                    "n_folds_valid"     : len(fold_metrics),
                    "accuracy"          : agg_metrics["accuracy"],
                    "macro_f1"          : agg_metrics["macro_f1"],
                    "recall_bahaya"     : agg_metrics["recall_bahaya"],
                    "false_alarm_rate"  : agg_metrics["false_alarm_rate"],
                    "miss_bahaya"       : agg_metrics["miss_bahaya"],
                }

                if ENABLE_YAWDD:
                    seg_lists = yawdd_scores.get(wcfg["name"], [])
                    total_segments = 0
                    false_alarm_segments = 0
                    session_alerts = 0

                    for seg_scores in seg_lists:
                        if not seg_scores:
                            continue
                        preds = [classify_score(s, tcfg["thresholds"],
                                                critical_override=use_override)
                                 for s in seg_scores]
                        total_segments += len(preds)
                        false_alarm_segments += sum(1 for p in preds if p > 0)
                        if any(p > 0 for p in preds):
                            session_alerts += 1

                    yawdd_sessions = len(seg_lists)
                    fp_session = (session_alerts / yawdd_sessions) if yawdd_sessions > 0 else 0.0
                    fa_rate = (false_alarm_segments / total_segments) if total_segments > 0 else 0.0

                    row.update({
                        "yawdd_sessions"        : yawdd_sessions,
                        "yawdd_fp_session"      : round(fp_session * 100, 2),
                        "yawdd_false_alarm_rate": round(fa_rate * 100, 2),
                    })
                all_results.append(row)

                if combo_idx % 10 == 0 or combo_idx == n_combos:
                    print(f"  [{combo_idx:3d}/{n_combos}] {wcfg['name']:<15} "
                          f"{tcfg['name']:<14} {override_name} | "
                          f"F1={row['macro_f1']:5.1f}% Rec_Bahaya={row['recall_bahaya']:5.1f}%")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3: SIMPAN HASIL
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_save_results(all_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Main CSV ──────────────────────────────────────────────────────────────
    main_csv = RESULTS_DIR / "level_validation_results.csv"
    if all_results:
        fields = list(all_results[0].keys())
        with open(main_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n[SAVED] {main_csv}")

    # ── Threshold Sensitivity Table (W3_Proposed only, no override) ───────────
    ts_rows = [r for r in all_results
               if r["weight_config"] == "W3_Proposed"
               and not r["critical_override"]]
    ts_csv = RESULTS_DIR / "threshold_sensitivity_table.csv"
    if ts_rows:
        fields_ts = ["threshold_config", "t_hati_hati", "t_waspada", "t_bahaya",
                     "accuracy", "macro_f1", "recall_bahaya", "false_alarm_rate"]
        with open(ts_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields_ts, extrasaction='ignore')
            w.writeheader()
            w.writerows(ts_rows)
        print(f"[SAVED] {ts_csv}")

        print(f"\n--- Threshold Sensitivity (W3_Proposed, No Override) ---")
        print(f"{'Threshold':<16} {'Acc':>7} {'MacroF1':>9} {'RecBahaya':>11} {'FalseAlarm':>12}")
        print("-" * 60)
        for r in ts_rows:
            marker = " <-- sistem" if r["threshold_config"] == "T2_30_50_70" else ""
            print(f"{r['threshold_config']:<16} {r['accuracy']:>6.1f}% "
                  f"{r['macro_f1']:>8.1f}% {r['recall_bahaya']:>10.1f}% "
                  f"{r['false_alarm_rate']:>11.1f}%{marker}")

    # ── Weight Ablation Table (T2 only, no override) ──────────────────────────
    wa_rows = [r for r in all_results
               if r["threshold_config"] == "T2_30_50_70"
               and not r["critical_override"]]
    wa_csv = RESULTS_DIR / "weight_ablation_table.csv"
    if wa_rows:
        fields_wa = ["weight_config", "w_eye", "w_mouth", "w_head",
                     "accuracy", "macro_f1", "recall_bahaya", "false_alarm_rate"]
        with open(wa_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields_wa, extrasaction='ignore')
            w.writeheader()
            w.writerows(wa_rows)
        print(f"[SAVED] {wa_csv}")

        print(f"\n--- Weight Ablation (T2=30/50/70, No Override) ---")
        print(f"{'Config':<16} {'Eye':>5} {'Mouth':>6} {'Head':>5} "
              f"{'Acc':>7} {'MacroF1':>9} {'RecBahaya':>11} {'FalseAlarm':>12}")
        print("-" * 78)
        for r in wa_rows:
            marker = " <-- usulan" if r["weight_config"] == "W3_Proposed" else ""
            print(f"{r['weight_config']:<16} {r['w_eye']:>5.2f} {r['w_mouth']:>6.2f} "
                  f"{r['w_head']:>5.2f} {r['accuracy']:>6.1f}% "
                  f"{r['macro_f1']:>8.1f}% {r['recall_bahaya']:>10.1f}% "
                  f"{r['false_alarm_rate']:>11.1f}%{marker}")

    # ── Critical Override Comparison (W3 + T2 only) ───────────────────────────
    ov_rows = [r for r in all_results
               if r["weight_config"] == "W3_Proposed"
               and r["threshold_config"] == "T2_30_50_70"]
    ov_csv = RESULTS_DIR / "critical_override_comparison.csv"
    if ov_rows:
        with open(ov_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(ov_rows[0].keys()))
            w.writeheader()
            w.writerows(ov_rows)
        print(f"[SAVED] {ov_csv}")

        print(f"\n--- Critical Override (W3_Proposed + T2=30/50/70) ---")
        for r in ov_rows:
            print(f"  {r['override_mode']:<18}: "
                  f"Acc={r['accuracy']:5.1f}% F1={r['macro_f1']:5.1f}% "
                  f"RecBahaya={r['recall_bahaya']:5.1f}% FA={r['false_alarm_rate']:5.1f}%")

    # Temukan konfigurasi terbaik (Macro F1 tertinggi, no override)
    best_rows = [r for r in all_results if not r["critical_override"]]
    if best_rows:
        best = max(best_rows, key=lambda r: r["macro_f1"])
        print(f"\n{'='*65}")
        print(f"KONFIGURASI TERBAIK (Macro F1 tertinggi, tanpa override):")
        print(f"  Weight   : {best['weight_config']} "
              f"({best['w_eye']}/{best['w_mouth']}/{best['w_head']})")
        print(f"  Threshold: {best['threshold_config']} "
              f"({best['t_hati_hati']}/{best['t_waspada']}/{best['t_bahaya']})")
        print(f"  Accuracy : {best['accuracy']:.1f}%")
        print(f"  Macro F1 : {best['macro_f1']:.1f}%")
        print(f"  Recall Bahaya: {best['recall_bahaya']:.1f}%")
        print(f"  False Alarm : {best['false_alarm_rate']:.1f}%")
        if "yawdd_fp_session" in best:
            print(f"  YawDD FP/Session: {best['yawdd_fp_session']:.2f}%")
            print(f"  YawDD False Alarm: {best['yawdd_false_alarm_rate']:.2f}%")
        print(f"{'='*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Threshold Level Validation — LOVO")
    parser.add_argument('--skip-gen',  action='store_true',
                        help='Skip Fase 1 (gunakan cache _perclos.csv yang ada)')
    parser.add_argument('--workers',   type=int, default=4,
                        help='Jumlah parallel workers Fase 1 (default: 4)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Hanya Fase 2+3 (alias untuk --skip-gen)')
    args = parser.parse_args()

    print("=" * 65)
    print("THRESHOLD LEVEL VALIDATION — LOVO")
    print(f"  Dataset    : 15 video DMD")
    print(f"  Segment    : {SEGMENT_SEC} detik")
    print(f"  Weight cfg : {len([w for w in WEIGHT_CONFIGS if w['name'] in SELECTED_WEIGHTS])}")
    print(f"  Threshold  : {len(THRESHOLD_CONFIGS)}")
    print(f"  Override   : 2 mode (A/B)")
    print(f"  AccOverride: {ACC_OVERRIDE_NAME if ACC_OVERRIDE else 'None'}")
    print(f"  YawDD Eval : {'On' if ENABLE_YAWDD else 'Off'}")
    total_eval = len([w for w in WEIGHT_CONFIGS if w['name'] in SELECTED_WEIGHTS]) * len(THRESHOLD_CONFIGS) * 2 * 15
    print(f"  Total eval : {total_eval} fold-evaluasi")
    print("=" * 65)

    t_total = time.time()

    # ── Pastikan GT labels sudah ada ──────────────────────────────────────────
    missing_gt = [i for i in range(1, 16)
                  if not (GT_DIR / f"gt_labels_ds{i:02d}.json").exists()]
    if missing_gt:
        print(f"\n[INFO] GT labels belum ada untuk DS: {missing_gt}")
        print("[INFO] Menjalankan gt_label_builder.py...")
        result = subprocess.run(
            [VENV_PYTHON, str(SCRIPT_DIR / "gt_label_builder.py")],
            cwd=str(SCRIPT_DIR)
        )
        if result.returncode != 0:
            print("[FATAL] gt_label_builder.py gagal. Hentikan.")
            sys.exit(1)
        print()

    # ── Fase 1: Score Generation ──────────────────────────────────────────────
    if not args.skip_gen and not args.eval_only:
        errors = phase1_generate_scores(max_workers=args.workers)
        if errors > len(WEIGHT_CONFIGS) * 15 * 0.3:
            print(f"[WARN] {errors} task gagal (> 30%). Cek _lovo_worker.py")
        phase1_generate_scores_yawdd(max_workers=args.workers)
    else:
        print("\n[SKIP] Fase 1 dilewati (--skip-gen). Pakai cache yang ada.")

    # ── Fase 2: LOVO Evaluation ───────────────────────────────────────────────
    all_results = phase2_lovo_evaluation()

    if not all_results:
        print("\n[ERROR] Tidak ada hasil evaluasi. "
              "Pastikan GT labels dan _perclos.csv sudah ada.")
        sys.exit(1)

    # ── Fase 3: Simpan hasil ──────────────────────────────────────────────────
    phase3_save_results(all_results)

    print(f"\nTotal runtime: {(time.time()-t_total)/60:.1f} menit")
    print("[DONE] Eksperimen selesai!")


if __name__ == '__main__':
    main()
