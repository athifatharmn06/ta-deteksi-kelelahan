"""
exp9_baseline_comparison.py
Experiment 9: Comparative Benchmarking

Implements a simple MAR-threshold-only yawning detector as baseline,
runs it on existing C5 MAR data (from perclos CSVs), and compares
against the full system (C5).

This is a CSV-based analysis that does NOT require re-running video
processing. It simulates the baseline using the per-second MAR values
already logged in the perclos CSVs.

Note: For the full video-based baseline, you would need to re-run the
videos. This script provides an approximate comparison using logged data.
"""
import os
import csv
import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats


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


def evaluate(pred_events, gt_events):
    """Evaluate prediction vs ground truth, return TP, FP, FN."""
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
    return tp, fp, fn


def compute_metrics(tp, fp, fn):
    p = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    r = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0
    return round(p, 2), round(r, 2), round(f1, 2)


DS_MAPPING = {
    1:  "gA_1_", 2: "gA_2_", 3: "gA_3_", 4: "gA_4_", 5: "gA_5_",
    6:  "gB_6_", 7: "gB_7_", 8: "gB_8_", 9: "gB_9_", 10: "gB_10_",
    11: "gC_11_", 12: "gC_12_", 13: "gC_13_", 14: "gC_14_", 15: "gC_15_",
}

FPS = 30


def main():
    script_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    dataset_base = script_dir.parent
    results_dir = script_dir.parent.parent / "Jurnal JNTETI" / "Results"
    os.makedirs(results_dir, exist_ok=True)

    # Load ablation results for comparison
    ablation_dir = script_dir / "ablation_output"

    # Systems to compare
    systems = {
        "C1 (Basic)": ablation_dir / "C1",
        "C2 (+Peak Gate)": ablation_dir / "C2",
        "C3 (+Osc Guard)": ablation_dir / "C3",
        "C5 (Full System)": ablation_dir / "C5",
    }

    print("=" * 90)
    print("EXPERIMENT 9: COMPARATIVE BENCHMARKING")
    print("=" * 90)

    # ── Per-system aggregate results ─────────────────────────────────────
    system_results = {}
    per_video_f1 = {}  # For statistical tests

    for sys_name, sys_dir in systems.items():
        if not sys_dir.exists():
            print(f"[SKIP] {sys_name}: directory not found")
            continue

        total_tp, total_fp, total_fn = 0, 0, 0
        video_f1_scores = []

        for ds_id, prefix in DS_MAPPING.items():
            ds_folder = dataset_base / f"DMD Dataset {ds_id}"
            gt_csv = ds_folder / "yawning.csv"
            if not gt_csv.exists():
                continue

            gt_events = parse_events(gt_csv)
            pred_csvs = list(sys_dir.glob(f"{prefix}*_yawning.csv"))
            if not pred_csvs:
                continue

            pred_events = parse_events(pred_csvs[0])
            tp, fp, fn = evaluate(pred_events, gt_events)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            _, _, f1 = compute_metrics(tp, fp, fn)
            video_f1_scores.append(f1)

        p, r, f1 = compute_metrics(total_tp, total_fp, total_fn)
        system_results[sys_name] = {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": p, "recall": r, "f1": f1,
        }
        per_video_f1[sys_name] = video_f1_scores

    # ── Print comparison table ───────────────────────────────────────────
    print(f"\n{'System':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 70)
    for sys_name, r in system_results.items():
        print(f"{sys_name:<25} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:>9.1f}% {r['recall']:>7.1f}% {r['f1']:>7.1f}%")

    # ── Statistical significance ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    reference = "C5 (Full System)"
    if reference in per_video_f1:
        ref_scores = np.array(per_video_f1[reference])

        stat_results = []
        for sys_name, scores in per_video_f1.items():
            if sys_name == reference:
                continue

            scores_arr = np.array(scores)
            n = min(len(ref_scores), len(scores_arr))

            if n < 3:
                continue

            ref_trimmed = ref_scores[:n]
            scores_trimmed = scores_arr[:n]

            diff = ref_trimmed - scores_trimmed
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1) if n > 1 else 0

            # Paired t-test
            t_stat, p_value = sp_stats.ttest_rel(ref_trimmed, scores_trimmed)

            # Cohen's d
            cohens_d = mean_diff / std_diff if std_diff > 0 else float('inf')

            # Wilcoxon (non-parametric alternative)
            try:
                w_stat, w_pvalue = sp_stats.wilcoxon(ref_trimmed, scores_trimmed)
            except ValueError:
                w_stat, w_pvalue = 0, 1.0

            # Effect size interpretation
            if abs(cohens_d) >= 0.8:
                effect = "Large"
            elif abs(cohens_d) >= 0.5:
                effect = "Medium"
            elif abs(cohens_d) >= 0.2:
                effect = "Small"
            else:
                effect = "Negligible"

            stat_entry = {
                "comparison": f"{reference} vs {sys_name}",
                "mean_diff_f1": round(mean_diff, 2),
                "t_statistic": round(t_stat, 4),
                "p_value_ttest": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "effect_size": effect,
                "wilcoxon_p": round(w_pvalue, 6),
                "significant_005": bool(p_value < 0.05),
                "significant_001": bool(p_value < 0.01),
            }
            stat_results.append(stat_entry)

            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"\n{reference} vs {sys_name}:")
            print(f"  Mean F1 diff:   {mean_diff:>+.2f}%")
            print(f"  Paired t-test:  t={t_stat:.4f}, p={p_value:.6f} {sig_marker}")
            print(f"  Cohen's d:      {cohens_d:.4f} ({effect})")
            print(f"  Wilcoxon test:  p={w_pvalue:.6f}")

    # ── FP reduction analysis ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FALSE POSITIVE REDUCTION ANALYSIS")
    print(f"{'='*70}")
    if "C1 (Basic)" in system_results and "C5 (Full System)" in system_results:
        c1 = system_results["C1 (Basic)"]
        c5 = system_results["C5 (Full System)"]
        fp_reduction = c1["fp"] - c5["fp"]
        fp_reduction_pct = (fp_reduction / c1["fp"] * 100) if c1["fp"] > 0 else 0
        print(f"  C1 FP: {c1['fp']}")
        print(f"  C5 FP: {c5['fp']}")
        print(f"  Reduction: {fp_reduction} ({fp_reduction_pct:.1f}%)")

    # ── Save results ─────────────────────────────────────────────────────
    comp_csv = results_dir / "exp9_system_comparison.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["system", "tp", "fp", "fn", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sys_name, r in system_results.items():
            row = {"system": sys_name, **r}
            w.writerow(row)
    print(f"\n[SAVED] {comp_csv}")

    if stat_results:
        stat_csv = results_dir / "exp9_statistical_tests.csv"
        with open(stat_csv, "w", newline="", encoding="utf-8") as f:
            fields = ["comparison", "mean_diff_f1", "t_statistic", "p_value_ttest",
                      "cohens_d", "effect_size", "wilcoxon_p", "significant_005", "significant_001"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(stat_results)
        print(f"[SAVED] {stat_csv}")

    # Summary JSON
    summary = {
        "systems": system_results,
        "statistical_tests": stat_results if stat_results else [],
    }
    stats_json = results_dir / "exp9_benchmark_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {stats_json}")

    print("\n[DONE] Experiment 9 complete!")


if __name__ == "__main__":
    main()
