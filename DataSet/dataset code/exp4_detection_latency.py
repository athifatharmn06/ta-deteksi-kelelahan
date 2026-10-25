"""
exp4_detection_latency.py
Experiment 4: Time-to-Detection Latency Analysis

Computes the delay between ground truth yawn start and system detection start
for all matched yawn events across 15 DMD s5 datasets.

Outputs:
  - Per-event latency table (CSV)
  - Aggregate statistics (console + saved to results)
  - Data for histogram plotting
"""
import os
import csv
import json
import numpy as np
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_events(csv_path, start_col="frame_start", end_col="frame_end"):
    """Read events from a CSV, return list of (frame_start, frame_end)."""
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


def parse_gt_events_detailed(csv_path):
    """Read GT yawning events with action_type metadata."""
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
    """Check if two frame ranges overlap."""
    return max(a_start, b_start) <= min(a_end, b_end)


def overlap_amount(a_start, a_end, b_start, b_end):
    """Compute the overlap in frames between two ranges."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


# ── Dataset mapping ─────────────────────────────────────────────────────────
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


def find_pred_csv(c5_dir, prefix):
    """Find a prediction yawning CSV matching the given prefix."""
    matches = list(Path(c5_dir).glob(f"{prefix}*_yawning.csv"))
    return matches[0] if matches else None


def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    c5_dir = script_dir / "ablation_output" / "C5"
    dataset_base = script_dir.parent  # c:\TA Deteksi Kelelahan\DataSet
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    os.makedirs(results_dir, exist_ok=True)

    all_latencies = []
    fn_cases = []          # For Exp 7 (will be saved separately)
    per_video_summary = []

    for ds_id, info in DS_MAPPING.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        gt_csv = ds_folder / "yawning.csv"

        if not gt_csv.exists():
            print(f"[SKIP] DS{ds_id:02d}: GT not found")
            continue

        pred_csv = find_pred_csv(c5_dir, info["prefix"])
        if pred_csv is None:
            print(f"[SKIP] DS{ds_id:02d}: Prediction not found")
            continue

        gt_events = parse_gt_events_detailed(gt_csv)
        pred_events = parse_events(pred_csv)

        # Match GT → Pred (greedy: each GT matched to best-overlapping pred)
        matched_gt = set()
        matched_pred = set()
        matches = []

        for g_idx, gt in enumerate(gt_events):
            best_overlap = 0
            best_p_idx = -1
            for p_idx, (ps, pe) in enumerate(pred_events):
                ov = overlap_amount(gt["frame_start"], gt["frame_end"], ps, pe)
                if ov > best_overlap:
                    best_overlap = ov
                    best_p_idx = p_idx

            if best_overlap > 0:
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)
                ps, pe = pred_events[best_p_idx]
                latency_frames = ps - gt["frame_start"]
                latency_sec = latency_frames / FPS

                gt_duration = (gt["frame_end"] - gt["frame_start"]) / FPS
                pred_duration = (pe - ps) / FPS

                entry = {
                    "dataset": f"DS{ds_id:02d}",
                    "group": info["group"],
                    "gt_frame_start": gt["frame_start"],
                    "gt_frame_end": gt["frame_end"],
                    "gt_duration_sec": round(gt_duration, 3),
                    "pred_frame_start": ps,
                    "pred_frame_end": pe,
                    "pred_duration_sec": round(pred_duration, 3),
                    "latency_frames": latency_frames,
                    "latency_sec": round(latency_sec, 3),
                    "action_type": gt["action_type"],
                }
                all_latencies.append(entry)
                matches.append(entry)

        # Identify FNs (unmatched GT events) for Exp 7
        for g_idx, gt in enumerate(gt_events):
            if g_idx not in matched_gt:
                fn_cases.append({
                    "dataset": f"DS{ds_id:02d}",
                    "group": info["group"],
                    "gt_frame_start": gt["frame_start"],
                    "gt_frame_end": gt["frame_end"],
                    "gt_duration_sec": round((gt["frame_end"] - gt["frame_start"]) / FPS, 3),
                    "action_type": gt["action_type"],
                    "reason": "unmatched",
                })

        # Per-video summary
        tp = len(matched_gt)
        fp = len(pred_events) - len(matched_pred)
        fn = len(gt_events) - len(matched_gt)
        latencies_this = [m["latency_sec"] for m in matches]
        per_video_summary.append({
            "dataset": f"DS{ds_id:02d}",
            "group": info["group"],
            "gt_count": len(gt_events),
            "pred_count": len(pred_events),
            "tp": tp, "fp": fp, "fn": fn,
            "mean_latency": round(np.mean(latencies_this), 3) if latencies_this else "N/A",
        })

        status = f"GT={len(gt_events)} Pred={len(pred_events)} TP={tp} FP={fp} FN={fn}"
        if latencies_this:
            status += f" AvgLat={np.mean(latencies_this):.2f}s"
        print(f"[DS{ds_id:02d}] {status}")

    # ── Aggregate Statistics ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: DETECTION LATENCY ANALYSIS")
    print("=" * 70)

    latency_values = [e["latency_sec"] for e in all_latencies]
    n = len(latency_values)

    if n == 0:
        print("[ERROR] No matched yawn events found!")
        return

    lat_arr = np.array(latency_values)
    mean_lat = np.mean(lat_arr)
    std_lat = np.std(lat_arr, ddof=1) if n > 1 else 0
    median_lat = np.median(lat_arr)
    min_lat = np.min(lat_arr)
    max_lat = np.max(lat_arr)
    within_1_5 = np.sum(lat_arr <= 1.5) / n * 100
    within_2_0 = np.sum(lat_arr <= 2.0) / n * 100
    within_3_0 = np.sum(lat_arr <= 3.0) / n * 100

    # 95% CI for mean
    if n > 1:
        from scipy import stats as sp_stats
        ci_low, ci_high = sp_stats.t.interval(0.95, df=n - 1, loc=mean_lat, scale=std_lat / np.sqrt(n))
    else:
        ci_low, ci_high = mean_lat, mean_lat

    print(f"\n{'Metric':<35} {'Value':>15}")
    print("-" * 55)
    print(f"{'N (matched yawns)':<35} {n:>15}")
    print(f"{'Mean latency (s)':<35} {mean_lat:>12.3f} ± {std_lat:.3f}")
    print(f"{'95% CI (s)':<35} [{ci_low:>6.3f}, {ci_high:.3f}]")
    print(f"{'Median latency (s)':<35} {median_lat:>15.3f}")
    print(f"{'Min latency (s)':<35} {min_lat:>15.3f}")
    print(f"{'Max latency (s)':<35} {max_lat:>15.3f}")
    print(f"{'% detected within 1.5s':<35} {within_1_5:>14.1f}%")
    print(f"{'% detected within 2.0s':<35} {within_2_0:>14.1f}%")
    print(f"{'% detected within 3.0s':<35} {within_3_0:>14.1f}%")

    # ── By Group ─────────────────────────────────────────────────────────
    print(f"\n{'Group':<8} {'N':>4} {'Mean±SD':>15} {'Median':>10} {'Range':>15}")
    print("-" * 55)
    for grp in ["gA", "gB", "gC"]:
        grp_lat = [e["latency_sec"] for e in all_latencies if e["group"] == grp]
        if grp_lat:
            g_arr = np.array(grp_lat)
            print(f"{grp:<8} {len(grp_lat):>4} {np.mean(g_arr):>7.3f}±{np.std(g_arr, ddof=1):.3f}"
                  f"   {np.median(g_arr):>10.3f} [{np.min(g_arr):.3f}, {np.max(g_arr):.3f}]")

    # ── By Action Type ───────────────────────────────────────────────────
    print(f"\n{'Action Type':<35} {'N':>4} {'Mean±SD':>15}")
    print("-" * 55)
    action_types = set(e["action_type"] for e in all_latencies)
    for at in sorted(action_types):
        at_lat = [e["latency_sec"] for e in all_latencies if e["action_type"] == at]
        if at_lat:
            a_arr = np.array(at_lat)
            print(f"{at:<35} {len(at_lat):>4} {np.mean(a_arr):>7.3f}±{np.std(a_arr, ddof=1) if len(at_lat)>1 else 0:.3f}")

    # ── Save CSVs ────────────────────────────────────────────────────────
    # 1. Per-event latency
    lat_csv = results_dir / "exp4_detection_latency_events.csv"
    with open(lat_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "group", "gt_frame_start", "gt_frame_end",
                  "gt_duration_sec", "pred_frame_start", "pred_frame_end",
                  "pred_duration_sec", "latency_frames", "latency_sec", "action_type"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_latencies)
    print(f"\n[SAVED] {lat_csv}")

    # 2. Per-video summary
    summary_csv = results_dir / "exp4_per_video_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "group", "gt_count", "pred_count", "tp", "fp", "fn", "mean_latency"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_video_summary)
    print(f"[SAVED] {summary_csv}")

    # 3. Aggregate statistics (JSON for easy consumption)
    stats_json = results_dir / "exp4_aggregate_stats.json"
    stats = {
        "n_matched": n,
        "mean_latency_sec": round(mean_lat, 4),
        "std_latency_sec": round(std_lat, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "median_latency_sec": round(median_lat, 4),
        "min_latency_sec": round(min_lat, 4),
        "max_latency_sec": round(max_lat, 4),
        "pct_within_1_5s": round(within_1_5, 2),
        "pct_within_2_0s": round(within_2_0, 2),
        "pct_within_3_0s": round(within_3_0, 2),
    }
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] {stats_json}")

    # 4. FN cases for Experiment 7
    fn_csv = results_dir / "exp7_false_negative_cases.csv"
    with open(fn_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "group", "gt_frame_start", "gt_frame_end",
                  "gt_duration_sec", "action_type", "reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(fn_cases)
    print(f"[SAVED] {fn_csv}  ({len(fn_cases)} false negatives)")

    print("\n[DONE] Experiment 4 complete!")


if __name__ == "__main__":
    main()
