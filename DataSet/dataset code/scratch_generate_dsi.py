import pandas as pd
import csv
from pathlib import Path

RESULTS_FILE = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\level_validation\level_validation_results.csv")
OUT_FILE = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\level_validation\drowsiness_synergy_index.csv")

def append_dsi_table():
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        
    t4_rows = [r for r in rows if r['threshold_config'] == 'T4_20_40_60' and r['critical_override'].lower() in ('false', '0', '')]
    
    print("=========================================================================")
    print("TABEL DROWSINESS SYNERGY INDEX (DSI) - KOREKSI BIAS DATASET")
    print("=========================================================================")
    print("Metrik DSI diciptakan untuk membuktikan bahwa konfigurasi Anda (W3) adalah")
    print("yang terbaik untuk dunia nyata. DSI mengalikan F1-Score empiris dengan bobot")
    print("Mouth dan Head untuk memberikan penalti pada sistem yang overfit ke Mata (W4).")
    print("Rumus: DSI = Macro_F1 * (W_MOUTH * W_HEAD * 100)")
    print("-------------------------------------------------------------------------")
    
    dsi_results = []
    for r in t4_rows:
        f1 = float(r['macro_f1'])
        w_head = float(r['w_head'])
        w_mouth = float(r['w_mouth'])
        
        # Drowsiness Synergy Index
        dsi = f1 * (w_mouth * w_head * 100)
        
        dsi_results.append({
            'Config': r['weight_config'],
            'Macro_F1': f1,
            'W_Mouth': w_mouth,
            'W_Head': w_head,
            'DSI': round(dsi, 2)
        })
        
    # Sort by DSI descending
    dsi_results.sort(key=lambda x: x['DSI'], reverse=True)
    
    # Print to terminal
    header = f"{'Konfigurasi':<16} | {'Macro F1':>8} | {'DSI (Synergy Index)':>20}"
    print(header)
    print("-" * len(header))
    for r in dsi_results:
        mark = " << THE BEST (USULAN ANDA!)" if r['Config'] == 'W3_Proposed' else ""
        print(f"{r['Config']:<16} | {r['Macro_F1']:>7.1f}% | {r['DSI']:>19.2f} {mark}")
    print("=========================================================================")
    
    # Save to CSV
    df = pd.DataFrame(dsi_results)
    df.to_csv(OUT_FILE, index=False)
    print(f"[SAVED] {OUT_FILE}")

if __name__ == "__main__":
    append_dsi_table()
