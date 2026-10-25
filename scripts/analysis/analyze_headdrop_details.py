import os
import csv
import numpy as np

BASE_DIR = r"c:\TA Deteksi Kelelahan\Laporan Progress\Pengujian 1 dan 2"
OUTPUT_DIR = os.path.join(BASE_DIR, "output csv")
ANOTASI_DIR = os.path.join(BASE_DIR, "anotasi")

subjects = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

print("| Subjek | Detik (Video) | Status | Min Pitch Dev | Masalah Utama |")
print("|---|---|---|---|---|")

for sub in subjects:
    gt_file = os.path.join(ANOTASI_DIR, sub, "headdrop_events.csv")
    raw_file = os.path.join(OUTPUT_DIR, sub, "raw_features.csv")
    
    if not os.path.exists(gt_file) or not os.path.exists(raw_file): continue
        
    gt_ts = []
    with open(gt_file, 'r') as f:
        r = csv.reader(f)
        next(r)
        for row in r: gt_ts.append(int(float(row[0])))
            
    raw_data = {}
    with open(raw_file, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(float(row['Timestamp_s']))
            if t not in raw_data: raw_data[t] = []
            raw_data[t].append(float(row['Pitch']))

    for g in gt_ts:
        # Check if detected (we can't easily check DT without loading it, but we know which ones are MISS from the report)
        # For now, let's just analyze the Pitch around GT
        peaks = []
        for t in range(g-3, g+4):
            if t in raw_data: peaks.extend(raw_data[t])
        
        if peaks:
            bp = np.mean(peaks[:10]) if len(peaks) > 10 else peaks[0]
            devs = [p - bp for p in peaks]
            min_dev = min(devs)
            
            # Simple check
            reason = "Pitch Dev < 10 deg" if min_dev > -10 else "Duration too short?"
            print(f"| {sub} | {g}s | Analyze | {min_dev:.2f} | {reason} |")
