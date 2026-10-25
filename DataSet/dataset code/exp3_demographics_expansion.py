"""
exp3_demographics_expansion.py
Experiment 3: Per-Class Demographics Expansion

Produces:
  1. Per-video detail table (DS01-DS15 individual results)
  2. Expanded demographic analysis by action_type (with/without hand)
  3. Per-group (gA, gB, gC) performance breakdown

Outputs CSV tables ready for paper import.
"""
import os
import csv
import json
import numpy as np
from pathlib import Path


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


def parse_gt_detailed(csv_path):
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fs = int(row["frame_start"])
                fe = int(row["frame_end"])
                action = row.get("action_type", "unknown")
                events.append({"frame_start": fs, "frame_end": fe, "action_type": action})
            except (ValueError, KeyError):
                pass
    return events


def overlaps(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) <= min(a_end, b_end)


DS_INFO = {
    1:  {"group": "gA", "subject": 1,  "prefix": "gA_1_",  "eth": "Caucasian", "glass": True,  "gender": "Male"},
    2:  {"group": "gA", "subject": 2,  "prefix": "gA_2_",  "eth": "Hispanic",  "glass": False, "gender": "Male"},
    3:  {"group": "gA", "subject": 3,  "prefix": "gA_3_",  "eth": "Hispanic",  "glass": False, "gender": "Male"},
    4:  {"group": "gA", "subject": 4,  "prefix": "gA_4_",  "eth": "Hispanic",  "glass": False, "gender": "Male"},
    5:  {"group": "gA", "subject": 5,  "prefix": "gA_5_",  "eth": "Hispanic",  "glass": False, "gender": "Male"},
    6:  {"group": "gB", "subject": 6,  "prefix": "gB_6_",  "eth": "Asian",     "glass": False, "gender": "Male"},
    7:  {"group": "gB", "subject": 7,  "prefix": "gB_7_",  "eth": "Hispanic",  "glass": False, "gender": "Male"},
    8:  {"group": "gB", "subject": 8,  "prefix": "gB_8_",  "eth": "Caucasian", "glass": False, "gender": "Male"},
    9:  {"group": "gB", "subject": 9,  "prefix": "gB_9_",  "eth": "Asian",     "glass": False, "gender": "Male"},
    10: {"group": "gB", "subject": 10, "prefix": "gB_10_", "eth": "Hispanic",  "glass": True,  "gender": "Male"},
    11: {"group": "gC", "subject": 11, "prefix": "gC_11_", "eth": "Hispanic",  "glass": True,  "gender": "Female"},
    12: {"group": "gC", "subject": 12, "prefix": "gC_12_", "eth": "Hispanic",  "glass": False, "gender": "Female"},
    13: {"group": "gC", "subject": 13, "prefix": "gC_13_", "eth": "Caucasian", "glass": False, "gender": "Female"},
    14: {"group": "gC", "subject": 14, "prefix": "gC_14_", "eth": "Hispanic",  "glass": False, "gender": "Male"},
    15: {"group": "gC", "subject": 15, "prefix": "gC_15_", "eth": "Hispanic",  "glass": False, "gender": "Female"},
}


def find_pred_csv(c5_dir, prefix):
    matches = list(Path(c5_dir).glob(f"{prefix}*_yawning.csv"))
    return matches[0] if matches else None


def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    c5_dir = script_dir / "ablation_output" / "C5"
    dataset_base = script_dir.parent
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    os.makedirs(results_dir, exist_ok=True)

    per_video_rows = []
    action_stats = {"with_hand": {"tp": 0, "fn": 0, "gt": 0},
                    "without_hand": {"tp": 0, "fn": 0, "gt": 0}}
    group_stats = {}
    eth_stats = {}
    glass_stats = {True: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "n": 0},
                   False: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "n": 0}}
    gender_stats = {}

    for ds_id, info in DS_INFO.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        gt_csv = ds_folder / "yawning.csv"
        pred_csv = find_pred_csv(c5_dir, info["prefix"])

        if not gt_csv.exists() or pred_csv is None:
            continue

        gt_events = parse_gt_detailed(gt_csv)
        pred_events = parse_events(pred_csv)

        # Match events
        matched_gt = set()
        matched_pred = set()
        for p_idx, (ps, pe) in enumerate(pred_events):
            for g_idx, gt in enumerate(gt_events):
                if overlaps(ps, pe, gt["frame_start"], gt["frame_end"]):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)

        tp = len(matched_gt)
        fp = len(pred_events) - len(matched_pred)
        fn = len(gt_events) - len(matched_gt)
        gt_count = len(gt_events)
        pred_count = len(pred_events)

        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        # Count by action type
        n_with_hand = sum(1 for g in gt_events if "with hand" in g["action_type"].lower()
                         and "without" not in g["action_type"].lower())
        n_without_hand = sum(1 for g in gt_events if "without hand" in g["action_type"].lower())

        # FN by action type
        fn_with_hand = sum(1 for g_idx, g in enumerate(gt_events)
                          if g_idx not in matched_gt
                          and "with hand" in g["action_type"].lower()
                          and "without" not in g["action_type"].lower())
        fn_without_hand = sum(1 for g_idx, g in enumerate(gt_events)
                             if g_idx not in matched_gt
                             and "without hand" in g["action_type"].lower())

        tp_with_hand = n_with_hand - fn_with_hand
        tp_without_hand = n_without_hand - fn_without_hand

        action_stats["with_hand"]["tp"] += tp_with_hand
        action_stats["with_hand"]["fn"] += fn_with_hand
        action_stats["with_hand"]["gt"] += n_with_hand
        action_stats["without_hand"]["tp"] += tp_without_hand
        action_stats["without_hand"]["fn"] += fn_without_hand
        action_stats["without_hand"]["gt"] += n_without_hand

        per_video_rows.append({
            "dataset": f"DS{ds_id:02d}",
            "subject": f"{info['group']}_{info['subject']}",
            "group": info["group"],
            "ethnicity": info["eth"],
            "glasses": "Yes" if info["glass"] else "No",
            "gender": info["gender"],
            "gt_total": gt_count,
            "gt_with_hand": n_with_hand,
            "gt_without_hand": n_without_hand,
            "predicted": pred_count,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 1),
            "recall": round(recall, 1),
            "f1": round(f1, 1),
        })

        # Group aggregation
        grp = info["group"]
        if grp not in group_stats:
            group_stats[grp] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "n": 0}
        group_stats[grp]["tp"] += tp
        group_stats[grp]["fp"] += fp
        group_stats[grp]["fn"] += fn
        group_stats[grp]["gt"] += gt_count
        group_stats[grp]["n"] += 1

        # Ethnicity aggregation
        eth = info["eth"]
        if eth not in eth_stats:
            eth_stats[eth] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "n": 0}
        eth_stats[eth]["tp"] += tp
        eth_stats[eth]["fp"] += fp
        eth_stats[eth]["fn"] += fn
        eth_stats[eth]["gt"] += gt_count
        eth_stats[eth]["n"] += 1

        # Glasses aggregation
        glass_stats[info["glass"]]["tp"] += tp
        glass_stats[info["glass"]]["fp"] += fp
        glass_stats[info["glass"]]["fn"] += fn
        glass_stats[info["glass"]]["gt"] += gt_count
        glass_stats[info["glass"]]["n"] += 1

        # Gender aggregation
        gen = info["gender"]
        if gen not in gender_stats:
            gender_stats[gen] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "n": 0}
        gender_stats[gen]["tp"] += tp
        gender_stats[gen]["fp"] += fp
        gender_stats[gen]["fn"] += fn
        gender_stats[gen]["gt"] += gt_count
        gender_stats[gen]["n"] += 1

    # ── Print results ────────────────────────────────────────────────────
    print("=" * 90)
    print("EXPERIMENT 3: PER-CLASS DEMOGRAPHICS EXPANSION")
    print("=" * 90)

    # Per-video table
    print(f"\n{'DS':<6} {'Subj':<8} {'Grp':<5} {'Eth':<10} {'Glass':<6} {'Gen':<6} "
          f"{'GT':>3} {'WH':>3} {'NH':>3} {'Pred':>4} {'TP':>3} {'FP':>3} {'FN':>3} "
          f"{'P%':>6} {'R%':>6} {'F1%':>6}")
    print("-" * 100)
    for r in per_video_rows:
        print(f"{r['dataset']:<6} {r['subject']:<8} {r['group']:<5} {r['ethnicity']:<10} "
              f"{r['glasses']:<6} {r['gender']:<6} {r['gt_total']:>3} {r['gt_with_hand']:>3} {r['gt_without_hand']:>3} "
              f"{r['predicted']:>4} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['precision']:>5.1f}% {r['recall']:>5.1f}% {r['f1']:>5.1f}%")

    # Action type breakdown
    print(f"\n{'Action Type':<25} {'GT':>4} {'TP':>4} {'FN':>4} {'Recall':>8}")
    print("-" * 50)
    for key, label in [("with_hand", "Yawning WITH hand"), ("without_hand", "Yawning WITHOUT hand")]:
        s = action_stats[key]
        recall = (s["tp"] / s["gt"] * 100) if s["gt"] > 0 else 0
        print(f"{label:<25} {s['gt']:>4} {s['tp']:>4} {s['fn']:>4} {recall:>7.1f}%")

    # Group breakdown
    print(f"\n{'Group':<8} {'n':>3} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
    print("-" * 60)
    for grp in ["gA", "gB", "gC"]:
        s = group_stats[grp]
        p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
        r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        print(f"{grp:<8} {s['n']:>3} {s['gt']:>4} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
              f"{p:>6.1f}% {r:>6.1f}% {f1:>6.1f}%")

    # Ethnicity breakdown
    print(f"\n{'Ethnicity':<12} {'n':>3} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
    print("-" * 60)
    for eth in ["Caucasian", "Hispanic", "Asian"]:
        if eth in eth_stats:
            s = eth_stats[eth]
            p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
            r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            print(f"{eth:<12} {s['n']:>3} {s['gt']:>4} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                  f"{p:>6.1f}% {r:>6.1f}% {f1:>6.1f}%")

    # Glasses breakdown
    print(f"\n{'Eyewear':<12} {'n':>3} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
    print("-" * 65)
    for glass, label in [(True, "Glasses"), (False, "No Glasses")]:
        s = glass_stats[glass]
        p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
        r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        print(f"{label:<12} {s['n']:>3} {s['gt']:>4} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
              f"{p:>6.1f}% {r:>6.1f}% {f1:>6.1f}%")

    # Gender breakdown
    print(f"\n{'Gender':<12} {'n':>3} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P%':>7} {'R%':>7} {'F1%':>7}")
    print("-" * 65)
    for gen in ["Male", "Female"]:
        if gen in gender_stats:
            s = gender_stats[gen]
            p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
            r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
            print(f"{gen:<12} {s['n']:>3} {s['gt']:>4} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4} "
                  f"{p:>6.1f}% {r:>6.1f}% {f1:>6.1f}%")

    # ── Save CSVs ────────────────────────────────────────────────────────
    # 1. Per-video detail
    detail_csv = results_dir / "exp3_per_video_detail.csv"
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "subject", "group", "ethnicity", "glasses", "gender",
                  "gt_total", "gt_with_hand", "gt_without_hand", "predicted",
                  "tp", "fp", "fn", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_video_rows)
    print(f"\n[SAVED] {detail_csv}")

    # 2. Action type breakdown
    action_csv = results_dir / "exp3_action_type_breakdown.csv"
    with open(action_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action_type", "gt", "tp", "fn", "recall_pct"])
        for key, label in [("with_hand", "Yawning WITH hand"), ("without_hand", "Yawning WITHOUT hand")]:
            s = action_stats[key]
            recall = (s["tp"] / s["gt"] * 100) if s["gt"] > 0 else 0
            w.writerow([label, s["gt"], s["tp"], s["fn"], round(recall, 1)])
    print(f"[SAVED] {action_csv}")

    # 3. Summary JSON
    summary = {
        "action_type_breakdown": {
            "with_hand": {**action_stats["with_hand"],
                         "recall_pct": round((action_stats["with_hand"]["tp"] / action_stats["with_hand"]["gt"] * 100) if action_stats["with_hand"]["gt"] > 0 else 0, 2)},
            "without_hand": {**action_stats["without_hand"],
                            "recall_pct": round((action_stats["without_hand"]["tp"] / action_stats["without_hand"]["gt"] * 100) if action_stats["without_hand"]["gt"] > 0 else 0, 2)},
        },
        "group_breakdown": {},
        "ethnicity_breakdown": {},
        "gender_breakdown": {},
    }
    for grp in ["gA", "gB", "gC"]:
        s = group_stats[grp]
        p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
        r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
        summary["group_breakdown"][grp] = {**s, "precision": round(p, 2), "recall": round(r, 2)}
    for eth, s in eth_stats.items():
        p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
        r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
        summary["ethnicity_breakdown"][eth] = {**s, "precision": round(p, 2), "recall": round(r, 2)}

    for gen, s in gender_stats.items():
        p = (s["tp"] / (s["tp"] + s["fp"]) * 100) if (s["tp"] + s["fp"]) > 0 else 0
        r = (s["tp"] / (s["tp"] + s["fn"]) * 100) if (s["tp"] + s["fn"]) > 0 else 0
        summary["gender_breakdown"][gen] = {**s, "precision": round(p, 2), "recall": round(r, 2)}

    stats_json = results_dir / "exp3_demographics_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {stats_json}")

    print("\n[DONE] Experiment 3 complete!")


if __name__ == "__main__":
    main()
