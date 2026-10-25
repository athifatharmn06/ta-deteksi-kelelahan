"""
Threshold sweep worker — runs a SINGLE video with specific thresholds.
This is a STANDALONE script called by exp2_threshold_sensitivity.py.
"""
import sys
import os
import json

video_path = sys.argv[1]
output_dir = sys.argv[2]
mar_threshold = float(sys.argv[3])
ear_threshold = float(sys.argv[4])

os.makedirs(output_dir, exist_ok=True)

# Import and configure BEFORE any other call
import v5_own_detection_model as mdl
mdl.USE_OSCILLATION_GUARD = True
mdl.USE_PEAK_GATE = False        # Disable for pure threshold sweep
mdl.USE_PERCLOS_WARMUP = True
mdl.USE_AUTO_CALIB_THRESH = False # Disable for Experiment 2 (Manual Override)

# Override threshold constants
# Note: THRESH_MAR_YAWN is the ABSOLUTE minimum MAR to consider yawning.
# The system also uses calibration to set a subject-specific threshold.
# We override BOTH the static threshold AND the calibration result.
mdl.THRESH_MAR_YAWN = mar_threshold
mdl.THRESH_EAR_CLOSE = ear_threshold

# Monkey-patch the DrowsinessDetector to use our threshold
_original_init = mdl.DrowsinessDetector.__init__

def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # Override the dynamically computed threshold with our sweep value
    # This works because THRESH_MAR_YAWN is used in update() but we can also set instance vars
    self.thresh_mar = mar_threshold
    # We must also override the global in the module because update() uses it
    mdl.THRESH_MAR_YAWN = mar_threshold
    mdl.THRESH_EAR_CLOSE = ear_threshold

mdl.DrowsinessDetector.__init__ = _patched_init

# Process
try:
    mdl.process_video(video_path, output_dir, headless=True)
    # Verify output was written
    from pathlib import Path
    csvs = list(Path(output_dir).glob("*_yawning.csv"))
    print(json.dumps({
        "status": "ok",
        "video": os.path.basename(video_path),
        "csvs": len(csvs),
    }), flush=True)
    sys.exit(0)
except SystemExit as e:
    # process_video may call sys.exit — check if CSVs were written anyway
    from pathlib import Path
    csvs = list(Path(output_dir).glob("*_yawning.csv"))
    if csvs:
        print(json.dumps({
            "status": "ok",
            "video": os.path.basename(video_path),
            "csvs": len(csvs),
            "note": f"SystemExit({e.code})"
        }), flush=True)
        os._exit(0)  # Force clean exit
    else:
        print(json.dumps({
            "status": "error",
            "video": os.path.basename(video_path),
            "error": f"SystemExit({e.code}), no CSVs written"
        }), flush=True)
        os._exit(1)
except Exception as e:
    print(json.dumps({
        "status": "error",
        "video": os.path.basename(video_path),
        "error": str(e)[:200]
    }), flush=True)
    sys.exit(1)
