"""
_lovo_worker.py
Worker subprocess untuk LOVO Score Generation.

Dipanggil oleh exp_level_validation.py per video per weight config.
Output: _perclos.csv (berisi score per detik) di output_dir.

Usage:
    python _lovo_worker.py <video_path> <output_dir> <w_eye> <w_mouth> <w_head> [<mar_thresh> <ear_thresh> <acc_override_json>]
"""

import sys
import os
import json
from pathlib import Path

# ── Parse args ────────────────────────────────────────────────────────────────
if len(sys.argv) < 6:
    print(json.dumps({"status": "error", "error": "Usage: _lovo_worker.py video output w_eye w_mouth w_head [mar ear acc_override_json]"}))
    sys.exit(1)

video_path  = sys.argv[1]
output_dir  = sys.argv[2]
w_eye       = float(sys.argv[3])
w_mouth     = float(sys.argv[4])
w_head      = float(sys.argv[5])
mar_thresh  = float(sys.argv[6]) if len(sys.argv) > 6 else 0.20
ear_thresh  = float(sys.argv[7]) if len(sys.argv) > 7 else 0.18
acc_override_json = sys.argv[8] if len(sys.argv) > 8 else None

os.makedirs(output_dir, exist_ok=True)

# ── Import & configure v5 model ──────────────────────────────────────────────
try:
    import v5_own_detection_model as mdl

    # Set bobot
    mdl.W_EYE   = w_eye
    mdl.W_MOUTH = w_mouth
    mdl.W_HEAD  = w_head

    # Set threshold sensor
    mdl.THRESH_MAR_YAWN  = mar_thresh
    mdl.THRESH_EAR_CLOSE = ear_thresh

    # Matikan auto-calibration threshold agar nilai kita berlaku
    mdl.USE_AUTO_CALIB_THRESH = False

    # Aktifkan semua fitur (C5 config = full system)
    mdl.USE_OSCILLATION_GUARD = True
    mdl.USE_PEAK_GATE         = True
    mdl.USE_PERCLOS_WARMUP    = True

    # Apply accumulator override if provided
    if acc_override_json:
        try:
            acc_override = json.loads(acc_override_json)
            mdl.ACC_OVERRIDE = acc_override.get("params") if isinstance(acc_override, dict) else None
            if isinstance(acc_override, dict) and "decay_scale_active" in acc_override:
                mdl.DECAY_SCALE_ACTIVE = float(acc_override["decay_scale_active"])
        except Exception:
            mdl.ACC_OVERRIDE = None

    # Jalankan
    result = mdl.process_video(video_path, output_dir, headless=True)

    print(json.dumps({
        "status"     : "ok",
        "video"      : os.path.basename(video_path),
        "output_dir" : output_dir,
        "w_eye"      : w_eye,
        "w_mouth"    : w_mouth,
        "w_head"     : w_head,
        "mar_thresh" : mar_thresh,
        "ear_thresh" : ear_thresh,
    }))

except Exception as e:
    import traceback
    print(json.dumps({
        "status"  : "error",
        "video"   : os.path.basename(video_path),
        "error"   : str(e),
        "trace"   : traceback.format_exc()[-500:],
    }))
    sys.exit(1)
