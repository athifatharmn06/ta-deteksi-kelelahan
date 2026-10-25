"""
v5_batch_eval.py — Headless batch evaluator
Menerima argumen: --out_dir, --guard, --peak, --warmup
Output: detection CSV per dataset di --out_dir
"""
import argparse, os, sys
from pathlib import Path

# Parse argumen
parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",    required=True)
parser.add_argument("--guard",      default="True")
parser.add_argument("--peak",       default="True")
parser.add_argument("--warmup",     default="True")
args = parser.parse_args()

# Set config flags SEBELUM import modul deteksi
import v5_own_detection_model as mdl
mdl.USE_OSCILLATION_GUARD = args.guard  == "True"
mdl.USE_PEAK_GATE         = args.peak   == "True"
mdl.USE_PERCLOS_WARMUP    = args.warmup == "True"

# Jalankan semua video di dataset 1-15
current_dir = Path(os.path.abspath(__file__)).parent
dataset_base = current_dir.parent 
print(f"[INFO] Mencari video di base: {dataset_base}")
video_files = []
for i in range(1, 16):
    folder = dataset_base / f"DMD Dataset {i}"
    if folder.exists():
        vids = list(folder.glob("*_ir_face.mp4"))
        if not vids:
             vids = list(folder.glob("*.mp4"))
        if vids:
             video_files.append(vids[0])
        else:
            print(f"  [WARNING] Tidak ada video MP4 di {folder.name}")
    else:
        print(f"  [WARNING] Folder tidak ditemukan: {folder.name}")

video_files.sort(key=lambda x: x.name)
print(f"[INFO] Ditemukan {len(video_files)} video untuk evaluasi.")

for vf in video_files:
    print(f"  Processing: {vf.name}")
    mdl.process_video(str(vf), args.out_dir, headless=True)

print("[DONE]")
