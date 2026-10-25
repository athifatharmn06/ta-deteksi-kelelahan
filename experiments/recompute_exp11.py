import os
import csv
from pathlib import Path

# Setup
script_dir = Path(r"C:\TA Deteksi Kelelahan\DataSet\dataset code")
sweep_base = script_dir / "weight_sweep_output"
dataset_base = script_dir.parent

DS_MAPPING = {
    1:  {"prefix": "gA_1_"},  2:  {"prefix": "gA_2_"},  3:  {"prefix": "gA_3_"},
    4:  {"prefix": "gA_4_"},  5:  {"prefix": "gA_5_"},  6:  {"prefix": "gB_6_"},
    7:  {"prefix": "gB_7_"},  8:  {"prefix": "gB_8_"},  9:  {"prefix": "gB_9_"},
    10: {"prefix": "gB_10_"}, 11: {"prefix": "gC_11_"}, 12: {"prefix": "gC_12_"},
    13: {"prefix": "gC_13_"}, 14: {"prefix": "gC_14_"}, 15: {"prefix": "gC_15_"},
}

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

WEIGHT_CONFIGS = ["Equal", "Proposed", "Alt1", "Alt2", "Alt3"]

for cfg in WEIGHT_CONFIGS:
    config_dir = sweep_base / cfg
    if not config_dir.exists(): continue
    
    total_tp, total_fp, total_fn = 0, 0, 0
    total_alerts = 0
    total_gt = 0
    
    for ds_id, info in DS_MAPPING.items():
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        # ONLY use yawning.csv (since normal blinks are NOT fatigue). 
        # Actually, also include microsleep if it exists in GT? DMD only drops perclos/yawning/blink in s5.
        yawn_gt = parse_events(ds_folder / "yawning.csv")
        gt_events = yawn_gt
        
        pred_csvs = list(config_dir.glob(f"{info['prefix']}*_events.csv"))
        if not pred_csvs: continue
        
        # In weight sweep, the alerts might be fatigue_alert or actually we can check all true fatigue outputs
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
    fpr = total_fp / total_alerts if total_alerts > 0 else 0
    fnr = total_fn / total_gt if total_gt > 0 else 0
    
    print(f"[{cfg}] TP={total_tp} FP={total_fp} FN={total_fn} Alerts={total_alerts} GT={total_gt}")
    print(f"  Prec={p*100:.1f}% Rec={r*100:.1f}% F1={f1*100:.2f}% FPR={fpr*100:.1f}%")
