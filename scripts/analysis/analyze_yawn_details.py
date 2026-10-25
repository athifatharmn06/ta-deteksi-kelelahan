import os
import csv
import numpy as np

BASE_DIR = r"c:\TA Deteksi Kelelahan\Laporan Progress\Pengujian 1 dan 2"
OUTPUT_DIR = os.path.join(BASE_DIR, "output csv")
ANOTASI_DIR = os.path.join(BASE_DIR, "anotasi")

subjects = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]

print("| Subjek | Detik (Video) | Status | Peak MAR | Peak MW Ratio | Masalah Utama |")
print("|---|---|---|---|---|---|")

for sub in subjects:
    gt_file = os.path.join(ANOTASI_DIR, sub, "yawn_events.csv")
    dt_file = os.path.join(OUTPUT_DIR, sub, "yawn_events.csv")
    raw_file = os.path.join(OUTPUT_DIR, sub, "raw_features.csv")
    
    if not os.path.exists(gt_file) or not os.path.exists(raw_file):
        continue
        
    gt_ts = []
    with open(gt_file, 'r') as f:
        r = csv.reader(f)
        next(r)
        for row in r: gt_ts.append(int(float(row[0])))
        
    dt_ts = []
    if os.path.exists(dt_file):
        with open(dt_file, 'r') as f:
            r = csv.reader(f)
            next(r)
            for row in r: dt_ts.append(int(float(row[0])))
            
    raw_data = {}
    with open(raw_file, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(float(row['Timestamp_s']))
            if t not in raw_data: raw_data[t] = []
            raw_data[t].append((float(row['MAR']), float(row['MW_Ratio'])))

    # Analisis FN (Miss)
    for g in gt_ts:
        matched = any(abs(g - d) <= 5 for d in dt_ts)
        if not matched:
            # Cari peak di sekitar timestamp
            peaks = []
            for t in range(g-3, g+4):
                if t in raw_data: peaks.extend(raw_data[t])
            
            if peaks:
                p_mar = max(p[0] for p in peaks)
                p_mw = max(p[1] for p in peaks)
                reason = "MAR < 0.45" if p_mar < 0.45 else "Terdeteksi Senyum (MW > 1.15)"
                print(f"| {sub} | {g}s | **MISS (FN)** | {p_mar:.2f} | {p_mw:.2f} | {reason} |")

    # Analisis FP (Salah Deteksi)
    for d in dt_ts:
        matched = any(abs(d - g) <= 5 for g in gt_ts)
        if not matched:
            peaks = []
            for t in range(d-3, d+4):
                if t in raw_data: peaks.extend(raw_data[t])
            if peaks:
                p_mar = max(p[0] for p in peaks)
                p_mw = max(p[1] for p in peaks)
                print(f"| {sub} | {d}s | **SALAH (FP)** | {p_mar:.2f} | {p_mw:.2f} | Mirip menguap tapi tawa/bicara |")
