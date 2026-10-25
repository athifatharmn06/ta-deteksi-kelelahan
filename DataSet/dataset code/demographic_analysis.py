
import os
import csv
from pathlib import Path

def get_events_from_csv(csv_path, start_col, end_col):
    events = []
    if not os.path.exists(csv_path):
        return events
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fs = int(row[start_col])
                fe = int(row[end_col])
                events.append((fs, fe))
            except:
                pass
    return events

def check_overlap(e1, e2):
    return max(e1[0], e2[0]) <= min(e1[1], e2[1])

def evaluate_video(pred_csv, gt_csv):
    gt_events = get_events_from_csv(gt_csv, "frame_start", "frame_end")
    pred_events = get_events_from_csv(pred_csv, "frame_start", "frame_end")
    
    matched_gt = set()
    matched_pred = set()
    
    for p_idx, p_ev in enumerate(pred_events):
        for g_idx, g_ev in enumerate(gt_events):
            if check_overlap(p_ev, g_ev):
                matched_gt.add(g_idx)
                matched_pred.add(p_idx)
                
    tp = len(matched_gt)
    fp = len(pred_events) - len(matched_pred)
    fn = len(gt_events) - len(matched_gt)
    return tp, fp, fn, len(gt_events), len(pred_events)

def main():
    # Use absolute paths
    current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    c5_dir = (current_dir / "ablation_output" / "C5").resolve()
    dataset_base = current_dir.parent.resolve()
    
    print(f"[INFO] C5 Dir: {c5_dir}")
    print(f"[INFO] Dataset Base: {dataset_base}")
    
    if not c5_dir.exists():
        print(f"[ERROR] C5 directory not found: {c5_dir}")
        return
    
    # Mapping DS -> Group
    mapping = {
        "DS01": {"eth": "Caucasian", "glass": True, "vid_prefix": "gA_1_"},
        "DS02": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gA_2_"},
        "DS03": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gA_3_"},
        "DS04": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gA_4_"},
        "DS05": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gA_5_"},
        "DS06": {"eth": "Asian",     "glass": False, "vid_prefix": "gB_6_"},
        "DS07": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gB_7_"},
        "DS08": {"eth": "Caucasian", "glass": False, "vid_prefix": "gB_8_"},
        "DS09": {"eth": "Asian",     "glass": False, "vid_prefix": "gB_9_"},
        "DS10": {"eth": "Hispanic",  "glass": True, "vid_prefix": "gB_10_"},
        "DS11": {"eth": "Hispanic",  "glass": True, "vid_prefix": "gC_11_"},
        "DS12": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gC_12_"},
        "DS13": {"eth": "Caucasian", "glass": False, "vid_prefix": "gC_13_"},
        "DS14": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gC_14_"},
        "DS15": {"eth": "Hispanic",  "glass": False, "vid_prefix": "gC_15_"},
    }
    
    stats_eth = {"Caucasian": {"tp":0, "fp":0, "fn":0, "gt":0, "n":0},
                 "Hispanic":  {"tp":0, "fp":0, "fn":0, "gt":0, "n":0},
                 "Asian":     {"tp":0, "fp":0, "fn":0, "gt":0, "n":0}}
                 
    stats_glass = {True:  {"tp":0, "fp":0, "fn":0, "gt":0, "n":0},
                   False: {"tp":0, "fp":0, "fn":0, "gt":0, "n":0}}
    
    for ds_name, info in mapping.items():
        ds_id = int(ds_name[2:])
        ds_folder = dataset_base / f"DMD Dataset {ds_id}"
        gt_csv = ds_folder / "yawning.csv"
        
        if not gt_csv.exists():
            print(f"[MISSING GT] {ds_name}: {gt_csv}")
            continue
            
        # Find pred file
        vids = list(c5_dir.glob(f"{info['vid_prefix']}*yawning.csv"))
        if not vids:
            print(f"[MISSING PRED] {ds_name} with prefix {info['vid_prefix']} in {c5_dir}")
            continue
            
        pred_csv = vids[0]
        tp, fp, fn, gt_len, pred_len = evaluate_video(pred_csv, gt_csv)
        print(f"[DEBUG] {ds_name}: GT_LEN={gt_len} PRED_LEN={pred_len} TP={tp} FP={fp} FN={fn}")
        
        # Aggregate
        eth = info['eth']
        stats_eth[eth]['tp'] += tp
        stats_eth[eth]['fp'] += fp
        stats_eth[eth]['fn'] += fn
        stats_eth[eth]['gt'] += gt_len
        stats_eth[eth]['n'] += 1
        
        glass = info['glass']
        stats_glass[glass]['tp'] += tp
        stats_glass[glass]['fp'] += fp
        stats_glass[glass]['fn'] += fn
        stats_glass[glass]['gt'] += gt_len
        stats_glass[glass]['n'] += 1
        
    print("\nDEMOGRAPHIC ANALYSIS - ETHNICITY")
    print(f"{'Ethnicity':<12} | {'n':<2} | {'GT':<2} | {'TP':<2} | {'FP':<2} | {'FN':<2} | {'Recall':<10}")
    print("-" * 60)
    for eth, s in stats_eth.items():
        recall = s['tp'] / s['gt'] if s['gt'] > 0 else 0
        print(f"{eth:<12} | {s['n']:<2} | {s['gt']:<2} | {s['tp']:<2} | {s['fp']:<2} | {s['fn']:<2} | {recall*100:>6.1f}%")
        
    print("\nDEMOGRAPHIC ANALYSIS - EYEWEAR")
    print(f"{'Eyewear':<12} | {'n':<2} | {'GT':<2} | {'TP':<2} | {'FP':<2} | {'FN':<2} | {'Recall':<10}")
    print("-" * 60)
    for glass, s in stats_glass.items():
        label = "Glasses" if glass else "No Glasses"
        recall = s['tp'] / s['gt'] if s['gt'] > 0 else 0
        print(f"{label:<12} | {s['n']:<2} | {s['gt']:<2} | {s['tp']:<2} | {s['fp']:<2} | {s['fn']:<2} | {recall*100:>6.1f}%")

if __name__ == "__main__":
    main()
