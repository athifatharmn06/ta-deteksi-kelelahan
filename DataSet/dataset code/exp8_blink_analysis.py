"""
exp8_blink_analysis.py
Experiment 8: Blink Detection Analysis

Compares automated blink detection (C5) against ground truth blink annotations
for all 15 DMD s5 datasets.

Outputs:
  - Per-video blink count comparison
  - Detection ratio analysis
  - Aggregate statistics
"""
import os
import csv
import json
import numpy as np
from pathlib import Path


def count_csv_rows(csv_path):
    """Count data rows in a CSV (excluding header)."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def parse_events(csv_path, start_col="frame_start", end_col="frame_end"):
    """Read events from CSV, return list of (frame_start, frame_end)."""
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


def overlaps(a, b, tolerance=5):
    """Check if two blink events overlap (with tolerance of ±5 frames)."""
    return max(a[0] - tolerance, b[0]) <= min(a[1] + tolerance, b[1])


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


def find_pred_csv(c5_dir, prefix, event_type="blink"):
    """Find prediction CSV matching prefix."""
    matches = list(Path(c5_dir).glob(f"{prefix}*_{event_type}.csv"))
    return matches[0] if matches else None


def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    c5_dir = script_dir / "ablation_output" / "C5"
    dataset_base = script_dir.parent
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    os.makedirs(results_dir, exist_ok=True)

    per_video_results = []
    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("=" * 80)
    print("EXPERIMENT 8: BLINK DETECTION ANALYSIS")
    print("=" * 80)
    print(f"\n{'DS':<6} {'Group':<6} {'GT':>5} {'Pred':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Ratio':>8} {'Status':<12}")
    print("-" * 80)

    for ds_id, info in DS_MAPPING.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        gt_csv = ds_folder / "blink.csv"
        pred_csv = find_pred_csv(c5_dir, info["prefix"], "blink")

        if not gt_csv.exists():
            print(f"DS{ds_id:02d}  {info['group']:<6} [GT blink.csv not found]")
            continue
        if pred_csv is None:
            print(f"DS{ds_id:02d}  {info['group']:<6} [Pred blink CSV not found]")
            continue

        gt_events = parse_events(gt_csv)
        pred_events = parse_events(pred_csv)

        gt_count = len(gt_events)
        pred_count = len(pred_events)

        # Match events with temporal tolerance
        matched_gt = set()
        matched_pred = set()

        for p_idx, p_ev in enumerate(pred_events):
            for g_idx, g_ev in enumerate(gt_events):
                if g_idx not in matched_gt and overlaps(p_ev, g_ev, tolerance=8):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    break

        tp = len(matched_gt)
        fp = pred_count - len(matched_pred)
        fn = gt_count - len(matched_gt)

        ratio = (pred_count / gt_count * 100) if gt_count > 0 else 0
        if ratio > 110:
            status = "OVER"
        elif ratio < 90:
            status = "UNDER"
        else:
            status = "ACCURATE"

        per_video_results.append({
            "dataset": f"DS{ds_id:02d}",
            "group": info["group"],
            "gt_blinks": gt_count,
            "pred_blinks": pred_count,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ratio_pct": round(ratio, 1),
            "status": status,
            "precision": round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0,
            "recall": round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0,
        })

        total_gt += gt_count
        total_pred += pred_count
        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(f"DS{ds_id:02d}  {info['group']:<6} {gt_count:>5} {pred_count:>5} "
              f"{tp:>5} {fp:>5} {fn:>5} {ratio:>7.1f}% {status:<12}")

    # ── Aggregates ───────────────────────────────────────────────────────
    print("-" * 80)
    total_ratio = (total_pred / total_gt * 100) if total_gt > 0 else 0
    total_precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0
    total_recall = (total_tp / (total_tp + total_fn) * 100) if (total_tp + total_fn) > 0 else 0
    total_f1 = (2 * total_precision * total_recall / (total_precision + total_recall)) if (total_precision + total_recall) > 0 else 0

    print(f"{'TOTAL':<12} {total_gt:>5} {total_pred:>5} "
          f"{total_tp:>5} {total_fp:>5} {total_fn:>5} {total_ratio:>7.1f}%")

    print(f"\nAggregate Metrics:")
    print(f"  Precision: {total_precision:.1f}%")
    print(f"  Recall:    {total_recall:.1f}%")
    print(f"  F1-Score:  {total_f1:.1f}%")

    # ── Category breakdown ───────────────────────────────────────────────
    n_over = sum(1 for r in per_video_results if r["status"] == "OVER")
    n_under = sum(1 for r in per_video_results if r["status"] == "UNDER")
    n_accurate = sum(1 for r in per_video_results if r["status"] == "ACCURATE")
    print(f"\nDetection Ratio Summary:")
    print(f"  Accurate (90-110%): {n_accurate}/{len(per_video_results)} videos")
    print(f"  Over-detection (>110%): {n_over}/{len(per_video_results)} videos")
    print(f"  Under-detection (<90%): {n_under}/{len(per_video_results)} videos")

    # ── Per group ────────────────────────────────────────────────────────
    print(f"\n{'Group':<8} {'GT':>6} {'Pred':>6} {'Ratio':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)
    for grp in ["gA", "gB", "gC"]:
        grp_gt = sum(r["gt_blinks"] for r in per_video_results if r["group"] == grp)
        grp_pred = sum(r["pred_blinks"] for r in per_video_results if r["group"] == grp)
        grp_tp = sum(r["tp"] for r in per_video_results if r["group"] == grp)
        grp_fp = sum(r["fp"] for r in per_video_results if r["group"] == grp)
        grp_fn = sum(r["fn"] for r in per_video_results if r["group"] == grp)
        grp_ratio = (grp_pred / grp_gt * 100) if grp_gt > 0 else 0
        grp_prec = (grp_tp / (grp_tp + grp_fp) * 100) if (grp_tp + grp_fp) > 0 else 0
        grp_rec = (grp_tp / (grp_tp + grp_fn) * 100) if (grp_tp + grp_fn) > 0 else 0
        print(f"{grp:<8} {grp_gt:>6} {grp_pred:>6} {grp_ratio:>7.1f}% {grp_prec:>9.1f}% {grp_rec:>7.1f}%")

    # ── Ratio statistics ─────────────────────────────────────────────────
    ratios = [r["ratio_pct"] for r in per_video_results]
    if ratios:
        r_arr = np.array(ratios)
        print(f"\nDetection Ratio Statistics:")
        print(f"  Mean:   {np.mean(r_arr):.1f}%")
        print(f"  Median: {np.median(r_arr):.1f}%")
        print(f"  Std:    {np.std(r_arr, ddof=1):.1f}%")
        print(f"  Range:  [{np.min(r_arr):.1f}%, {np.max(r_arr):.1f}%]")

    # ── Save results ─────────────────────────────────────────────────────
    blink_csv = results_dir / "exp8_blink_analysis.csv"
    with open(blink_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "group", "gt_blinks", "pred_blinks", "tp", "fp", "fn",
                  "ratio_pct", "status", "precision", "recall"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_video_results)
    print(f"\n[SAVED] {blink_csv}")

    # Aggregate stats JSON
    stats = {
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "aggregate_precision": round(total_precision, 2),
        "aggregate_recall": round(total_recall, 2),
        "aggregate_f1": round(total_f1, 2),
        "mean_ratio_pct": round(float(np.mean(ratios)), 2) if ratios else 0,
        "n_accurate": n_accurate,
        "n_over": n_over,
        "n_under": n_under,
    }
    stats_json = results_dir / "exp8_blink_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] {stats_json}")

    print("\n[DONE] Experiment 8 complete!")


if __name__ == "__main__":
    main()
