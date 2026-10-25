"""
exp7_error_analysis.py
Experiment 7: Error Analysis Deep Dive

Classifies all false negative yawning events by root cause.
Uses the FN cases from exp4_detection_latency.py output.

Root cause categories:
  - Hand occlusion (action_type contains "with hand")
  - Short duration (< 1.5 seconds)
  - Adjacent to other event
  - Unknown
"""
import os
import csv
import json
import numpy as np
from pathlib import Path


def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"

    fn_csv = results_dir / "exp7_false_negative_cases.csv"
    if not fn_csv.exists():
        print("[ERROR] Run exp4_detection_latency.py first to generate FN cases!")
        return

    # Load FN cases
    fn_cases = []
    with open(fn_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["gt_duration_sec"] = float(row["gt_duration_sec"])
            row["gt_frame_start"] = int(row["gt_frame_start"])
            row["gt_frame_end"] = int(row["gt_frame_end"])
            fn_cases.append(row)

    print("=" * 80)
    print("EXPERIMENT 7: ERROR ANALYSIS DEEP DIVE")
    print(f"Total False Negatives: {len(fn_cases)}")
    print("=" * 80)

    # ── Classify root causes ─────────────────────────────────────────────
    for fn in fn_cases:
        causes = []
        action = fn["action_type"].lower()

        # Hand occlusion
        if "with hand" in action and "without" not in action:
            causes.append("Hand Occlusion")

        # Short duration
        if fn["gt_duration_sec"] < 1.5:
            causes.append("Short Duration (<1.5s)")

        # Very short (sub-second)
        if fn["gt_duration_sec"] < 1.0:
            causes.append("Sub-second Event")

        # If no hand and normal duration → landmark issue or threshold
        if not causes:
            if fn["gt_duration_sec"] >= 3.0:
                causes.append("Suppressed Mouth Opening")
            else:
                causes.append("Borderline Duration")

        fn["root_cause"] = "; ".join(causes)
        fn["primary_cause"] = causes[0]

    # ── Summary statistics ───────────────────────────────────────────────
    # Count by primary cause
    cause_counts = {}
    for fn in fn_cases:
        pc = fn["primary_cause"]
        cause_counts[pc] = cause_counts.get(pc, 0) + 1

    print(f"\n{'Root Cause':<30} {'Count':>6} {'Percentage':>10}")
    print("-" * 50)
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        pct = count / len(fn_cases) * 100
        print(f"{cause:<30} {count:>6} {pct:>9.1f}%")

    # ── Duration distribution of FNs ─────────────────────────────────────
    durations = [fn["gt_duration_sec"] for fn in fn_cases]
    dur_arr = np.array(durations)
    print(f"\nFN Duration Statistics:")
    print(f"  Mean:   {np.mean(dur_arr):.2f}s")
    print(f"  Median: {np.median(dur_arr):.2f}s")
    print(f"  Min:    {np.min(dur_arr):.2f}s")
    print(f"  Max:    {np.max(dur_arr):.2f}s")
    print(f"  < 1.0s: {np.sum(dur_arr < 1.0)} ({np.sum(dur_arr < 1.0)/len(dur_arr)*100:.1f}%)")
    print(f"  < 1.5s: {np.sum(dur_arr < 1.5)} ({np.sum(dur_arr < 1.5)/len(dur_arr)*100:.1f}%)")
    print(f"  >= 3.0s: {np.sum(dur_arr >= 3.0)} ({np.sum(dur_arr >= 3.0)/len(dur_arr)*100:.1f}%)")

    # ── By action type ───────────────────────────────────────────────────
    with_hand = [fn for fn in fn_cases if "Hand Occlusion" in fn["root_cause"]]
    without_hand = [fn for fn in fn_cases if "Hand Occlusion" not in fn["root_cause"]]
    print(f"\nAction Type Breakdown:")
    print(f"  With Hand (occluded):    {len(with_hand)} ({len(with_hand)/len(fn_cases)*100:.1f}%)")
    print(f"  Without Hand:            {len(without_hand)} ({len(without_hand)/len(fn_cases)*100:.1f}%)")

    # ── By dataset ───────────────────────────────────────────────────────
    print(f"\n{'Dataset':<8} {'Count':>6} {'Hand Occ.':>10} {'Short':>6} {'Other':>6}")
    print("-" * 40)
    ds_counts = {}
    for fn in fn_cases:
        ds = fn["dataset"]
        if ds not in ds_counts:
            ds_counts[ds] = {"total": 0, "hand": 0, "short": 0, "other": 0}
        ds_counts[ds]["total"] += 1
        if "Hand Occlusion" in fn["root_cause"]:
            ds_counts[ds]["hand"] += 1
        elif "Short" in fn["root_cause"] or "Sub-second" in fn["root_cause"]:
            ds_counts[ds]["short"] += 1
        else:
            ds_counts[ds]["other"] += 1

    for ds in sorted(ds_counts.keys()):
        c = ds_counts[ds]
        print(f"{ds:<8} {c['total']:>6} {c['hand']:>10} {c['short']:>6} {c['other']:>6}")

    # ── Per-group ────────────────────────────────────────────────────────
    print(f"\n{'Group':<8} {'Total FN':>8} {'Hand Occ.':>10} {'Other':>6}")
    print("-" * 35)
    for grp in ["gA", "gB", "gC"]:
        grp_fns = [fn for fn in fn_cases if fn["group"] == grp]
        grp_hand = sum(1 for fn in grp_fns if "Hand Occlusion" in fn["root_cause"])
        print(f"{grp:<8} {len(grp_fns):>8} {grp_hand:>10} {len(grp_fns)-grp_hand:>6}")

    # ── Detailed table ───────────────────────────────────────────────────
    print(f"\n{'#':>3} {'DS':<6} {'Grp':<5} {'Start':>7} {'End':>7} {'Dur(s)':>8} {'Action Type':<30} {'Root Cause':<30}")
    print("-" * 100)
    for i, fn in enumerate(fn_cases, 1):
        action_short = fn["action_type"].split("/")[-1] if "/" in fn["action_type"] else fn["action_type"]
        print(f"{i:>3} {fn['dataset']:<6} {fn['group']:<5} {fn['gt_frame_start']:>7} "
              f"{fn['gt_frame_end']:>7} {fn['gt_duration_sec']:>8.3f} "
              f"{action_short:<30} {fn['root_cause']:<30}")

    # ── Save classified FNs ──────────────────────────────────────────────
    classified_csv = results_dir / "exp7_fn_classified.csv"
    with open(classified_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "group", "gt_frame_start", "gt_frame_end",
                  "gt_duration_sec", "action_type", "root_cause", "primary_cause"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(fn_cases)
    print(f"\n[SAVED] {classified_csv}")

    # Summary JSON
    summary = {
        "total_fn": len(fn_cases),
        "cause_distribution": cause_counts,
        "hand_occlusion_count": len(with_hand),
        "hand_occlusion_pct": round(len(with_hand) / len(fn_cases) * 100, 1),
        "non_hand_count": len(without_hand),
        "duration_stats": {
            "mean": round(float(np.mean(dur_arr)), 3),
            "median": round(float(np.median(dur_arr)), 3),
            "min": round(float(np.min(dur_arr)), 3),
            "max": round(float(np.max(dur_arr)), 3),
        }
    }
    stats_json = results_dir / "exp7_error_analysis_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {stats_json}")

    print("\n[DONE] Experiment 7 complete!")


if __name__ == "__main__":
    main()
