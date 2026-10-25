"""
run_fase1_direct.py
Jalankan process_video untuk semua 15 video x 5 weight config
LANGSUNG (bukan subprocess) — lebih reliable.
"""
import sys, os, time
from pathlib import Path

SCRIPT_DIR = Path(r"c:\TA Deteksi Kelelahan\DataSet\dataset code")
DATASET_BASE = Path(r"c:\TA Deteksi Kelelahan\DataSet")
SCORES_DIR = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\level_validation\scores")

sys.path.insert(0, str(SCRIPT_DIR))

DS_MAPPING = {i: f"DMD Dataset {i}" for i in range(1, 16)}

WEIGHT_CONFIGS = [
    {"name": "W1_Equal",    "weights": (0.33, 0.33, 0.34)},
    {"name": "W2_EyeHeavy", "weights": (0.50, 0.25, 0.25)},
    {"name": "W3_Proposed", "weights": (0.45, 0.25, 0.30)},
    {"name": "W4_MaxEye",   "weights": (0.60, 0.20, 0.20)},
    {"name": "W5_MouthUp",  "weights": (0.40, 0.30, 0.30)},
]

total = 0
skip  = 0
error = 0
t0    = time.time()

for wi, wcfg in enumerate(WEIGHT_CONFIGS):
    w_eye, w_mouth, w_head = wcfg["weights"]
    out_base = SCORES_DIR / wcfg["name"]
    out_base.mkdir(parents=True, exist_ok=True)

    for ds_id in range(1, 16):
        ds_folder = DATASET_BASE / DS_MAPPING[ds_id]
        vids = list(ds_folder.glob("*_ir_face.mp4")) or list(ds_folder.glob("*.mp4"))
        if not vids:
            print(f"  [SKIP] DS{ds_id:02d} tidak ada video")
            continue

        video_path = str(vids[0])
        video_stem = vids[0].stem
        perclos_out = out_base / f"{video_stem}_perclos.csv"

        if perclos_out.exists():
            print(f"  [CACHE] {wcfg['name']} DS{ds_id:02d}")
            skip += 1
            continue

        # Fresh import per konfigurasi untuk reset global state
        if "v5_own_detection_model" in sys.modules:
            del sys.modules["v5_own_detection_model"]

        import v5_own_detection_model as mdl
        mdl.W_EYE   = w_eye
        mdl.W_MOUTH = w_mouth
        mdl.W_HEAD  = w_head
        mdl.THRESH_MAR_YAWN  = 0.20
        mdl.THRESH_EAR_CLOSE = 0.18
        mdl.USE_AUTO_CALIB_THRESH = False
        mdl.USE_OSCILLATION_GUARD = True
        mdl.USE_PEAK_GATE         = True
        mdl.USE_PERCLOS_WARMUP    = True

        print(f"  [{wcfg['name']} DS{ds_id:02d}] processing...", end=" ", flush=True)
        t1 = time.time()
        try:
            result = mdl.process_video(video_path, str(out_base), headless=True)
            elapsed = time.time() - t1
            if result:
                total += 1
                print(f"ok ({elapsed:.0f}s)")
            else:
                error += 1
                print(f"FAILED")
        except Exception as e:
            error += 1
            print(f"ERROR: {e}")

        del mdl
        if "v5_own_detection_model" in sys.modules:
            del sys.modules["v5_own_detection_model"]

print(f"\n=== SELESAI: {total} ok, {skip} cache, {error} error | {(time.time()-t0)/60:.1f} menit ===")
