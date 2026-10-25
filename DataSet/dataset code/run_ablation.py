"""
run_ablation.py
Jalankan 5 konfigurasi ablation secara otomatis headless (tanpa GUI)
"""
import subprocess
import sys
import os

# 5 Konfigurasi ablation
CONFIGS = [
    {"name": "C1", "GUARD": "False", "PEAK": "False",  "WARMUP": "False"},
    {"name": "C2", "GUARD": "False", "PEAK": "True",   "WARMUP": "False"},
    {"name": "C3", "GUARD": "True",  "PEAK": "True",   "WARMUP": "False"},
    {"name": "C4", "GUARD": "True",  "PEAK": "True",   "WARMUP": "True"},
    {"name": "C5", "GUARD": "True",  "PEAK": "True",   "WARMUP": "True"},   # full system
]

SCRIPT = "v5_batch_eval.py"
OUT_DIR   = "ablation_output"

current_dir = os.path.dirname(os.path.abspath(__file__))

for cfg in CONFIGS:
    out_subdir = os.path.join(current_dir, OUT_DIR, cfg["name"])
    os.makedirs(out_subdir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Menjalankan {cfg['name']}: GUARD={cfg['GUARD']} PEAK={cfg['PEAK']} WARMUP={cfg['WARMUP']}")
    print(f"{'='*50}")

    cmd = [
        sys.executable, SCRIPT,
        "--out_dir",    out_subdir,
        "--guard",      cfg["GUARD"],
        "--peak",       cfg["PEAK"],
        "--warmup",     cfg["WARMUP"],
    ]
    subprocess.run(cmd, check=True, cwd=current_dir)

print("\n[DONE] Semua konfigurasi selesai!")
print(f"Cek folder: {OUT_DIR}/")
