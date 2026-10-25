import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from scipy.stats import pearsonr

def calculate_metrics(gt_df, sys_df):
    """Aligns two dataframes by 'second' and calculates R and MAE."""
    # Ensure 'second' is integer
    gt_df['second'] = gt_df['second'].astype(int)
    sys_df['second'] = sys_df['second'].astype(int)
    
    # Merge on second
    merged = pd.merge(gt_df[['second', 'perclos']], 
                     sys_df[['second', 'perclos']], 
                     on='second', suffixes=('_gt', '_sys'))
    
    if len(merged) < 2:
        return None
    
    # PERCLOS in GT is 0.0-1.0 meistens but sometimes 0-100. Normalize to 0-100.
    # Check if GT perclos is 0-1 range
    if merged['perclos_gt'].max() <= 1.05 and merged['perclos_gt'].max() > 0:
        merged['perclos_gt'] = merged['perclos_gt'] * 100.0
        
    # In my model, perclos is already 0-100
    
    y_gt = merged['perclos_gt'].values
    y_sys = merged['perclos_sys'].values
    
    r_val, _ = pearsonr(y_gt, y_sys)
    mae = np.mean(np.abs(y_gt - y_sys))
    
    return {
        "samples": len(merged),
        "correlation_r": round(float(r_val), 4),
        "mae": round(float(mae), 4),
        "gt_mean": round(float(np.mean(y_gt)), 4),
        "sys_mean": round(float(np.mean(y_sys)), 4)
    }

def main():
    dataset_dir = Path(r"c:\TA Deteksi Kelelahan\DataSet")
    results_dir = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results")
    os.makedirs(results_dir, exist_ok=True)
    
    subject_results = []
    
    for i in range(1, 16):
        folder = dataset_dir / f"DMD Dataset {i}"
        gt_file = folder / "perclos.csv"
        sys_file = folder / "own_detection_model_perclos.csv"
        
        if gt_file.exists() and sys_file.exists():
            print(f"[PROCESSING] Folder {folder.name}...")
            try:
                gt_df = pd.read_csv(gt_file)
                sys_df = pd.read_csv(sys_file)
                
                metrics = calculate_metrics(gt_df, sys_df)
                if metrics:
                    metrics['subject'] = i
                    subject_results.append(metrics)
                    print(f"  R={metrics['correlation_r']}, MAE={metrics['mae']}")
            except Exception as e:
                print(f"  [ERROR] {e}")
        else:
            print(f"[SKIP] Missing files in {folder.name}")
            
    if not subject_results:
        print("[FATAL] No data validated.")
        return
        
    # Save per-subject CSV
    res_df = pd.DataFrame(subject_results)
    csv_path = results_dir / "exp5_perclos_validation_results.csv"
    res_df.to_csv(csv_path, index=False)
    
    # Aggregate Summary
    summary = {
        "exp": "Exp 5: PERCLOS Validation (System vs DMD Ground Truth)",
        "total_subjects": len(subject_results),
        "metrics": {
            "mean_correlation_r": round(float(res_df['correlation_r'].mean()), 4),
            "mean_mae": round(float(res_df['mae'].mean()), 4),
            "max_mae": round(float(res_df['mae'].max()), 4),
            "min_correlation_r": round(float(res_df['correlation_r'].min()), 4)
        }
    }
    
    json_path = results_dir / "exp5_perclos_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[DONE] PERCLOS Validation complete.")
    print(f"  Overall Correlation (R): {summary['metrics']['mean_correlation_r']}")
    print(f"  Overall MAE: {summary['metrics']['mean_mae']}%")
    print(f"  Results saved to {results_dir}")

if __name__ == "__main__":
    main()
