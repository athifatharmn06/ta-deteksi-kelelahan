import sys, os, time
from pathlib import Path

SCRIPT_DIR = Path(r"c:\TA Deteksi Kelelahan\DataSet\dataset code")
DATASET_BASE = Path(r"c:\TA Deteksi Kelelahan\DataSet")
SCORES_DIR = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\level_validation\scores")

sys.path.insert(0, str(SCRIPT_DIR))

DS_MAPPING = {i: f"DMD Dataset {i}" for i in range(1, 16)}

ALL_CONFIGS = {
    "W1_Equal":    (0.33, 0.33, 0.34),
    "W2_EyeHeavy": (0.50, 0.25, 0.25),
    "W3_Proposed": (0.45, 0.25, 0.30),
    "W4_MaxEye":   (0.60, 0.20, 0.20),
    "W5_MouthUp":  (0.40, 0.30, 0.30),
}

if len(sys.argv) < 2 or sys.argv[1] not in ALL_CONFIGS:
    print(f"Usage: python run_fase1_direct_parallel.py [CONFIG_NAME]")
    print(f"Available configs: {list(ALL_CONFIGS.keys())}")
    sys.exit(1)

config_name = sys.argv[1]
w_eye, w_mouth, w_head = ALL_CONFIGS[config_name]

out_base = SCORES_DIR / config_name
out_base.mkdir(parents=True, exist_ok=True)

total = 0
skip  = 0
error = 0
t0    = time.time()

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
        print(f"  [CACHE] {config_name} DS{ds_id:02d}")
        skip += 1
        continue

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

    print(f"  [{config_name} DS{ds_id:02d}] processing...", end=" ", flush=True)
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

print(f"\n=== SELESAI {config_name}: {total} ok, {skip} cache, {error} error | {(time.time()-t0)/60:.1f} menit ===")
