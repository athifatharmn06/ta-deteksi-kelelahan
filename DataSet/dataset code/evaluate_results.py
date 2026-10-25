"""
evaluate_results.py
Menghitung Precision, Recall, dan F1 Score untuk 5 konfigurasi Ablation Study.
Membandingkan output CSV (ablation_output/CX/*_yawning.csv) dengan ground truth (DMD Dataset X/yawning.csv)
"""
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

def evaluate_config(config_dir, dataset_base):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Iterate through all 15 datasets
    for i in range(1, 16):
        ds_folder = dataset_base / f"DMD Dataset {i}"
        if not ds_folder.exists():
            continue
            
        gt_csv = ds_folder / "yawning.csv"
        gt_events = get_events_from_csv(gt_csv, "frame_start", "frame_end")
        
        # Find the prediction csv for this dataset
        # The prediction csv is named {source_name}_yawning.csv
        # We need to find the video name first to know the source_name
        vids = list(ds_folder.glob("*_ir_face.mp4"))
        if not vids:
             vids = list(ds_folder.glob("*.mp4"))
             
        if not vids:
            continue
            
        source_name = vids[0].stem
        pred_csv = config_dir / f"{source_name}_yawning.csv"
        pred_events = get_events_from_csv(pred_csv, "frame_start", "frame_end")
        
        # Match events
        matched_gt = set()
        matched_pred = set()
        
        for p_idx, p_ev in enumerate(pred_events):
            for g_idx, g_ev in enumerate(gt_events):
                if check_overlap(p_ev, g_ev):
                    matched_gt.add(g_idx)
                    matched_pred.add(p_idx)
                    
        total_tp += len(matched_gt)
        total_fp += len(pred_events) - len(matched_pred)
        total_fn += len(gt_events) - len(matched_gt)
        
    return total_tp, total_fp, total_fn

def main():
    current_dir = Path(__file__).parent
    ablation_dir = current_dir / "ablation_output"
    dataset_base = current_dir.parent  # c:\TA Deteksi Kelelahan\DataSet
    
    configs = ["C1", "C2", "C3", "C4", "C5"]
    
    print(f"{'Config':<8} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<8} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 70)
    
    results = []
    
    for cfg in configs:
        cfg_dir = ablation_dir / cfg
        if not cfg_dir.exists():
            print(f"{cfg:<8} | Folder not found")
            continue
            
        tp, fp, fn = evaluate_config(cfg_dir, dataset_base)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'cfg': cfg,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
        
        print(f"{cfg:<8} | {precision*100:>6.1f}%    | {recall*100:>5.1f}%   | {f1*100:>5.1f}%   | {tp:<4} | {fp:<4} | {fn:<4}")
        
    print("-" * 70)

if __name__ == "__main__":
    main()
