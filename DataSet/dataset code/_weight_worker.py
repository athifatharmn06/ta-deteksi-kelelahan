"""
_weight_worker.py
Weight sweep worker — runs a SINGLE video with specific fusion weights.
Uses v5_own_detection_model.process_video directly.
"""
import sys
import os
import json
import csv
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_worker():
    if len(sys.argv) < 8:
        print(json.dumps({"status": "error", "message": "Missing arguments"}))
        return

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    w_eye      = float(sys.argv[3])
    w_mouth    = float(sys.argv[4])
    w_head     = float(sys.argv[5])
    mar_th     = float(sys.argv[6])
    ear_th     = float(sys.argv[7])

    os.makedirs(output_dir, exist_ok=True)

    # Import and configure
    import v5_own_detection_model as mdl
    
    # Override globals
    mdl.W_EYE    = w_eye
    mdl.W_MOUTH  = w_mouth
    mdl.W_HEAD   = w_head
    mdl.THRESH_MAR_YAWN  = mar_th
    mdl.THRESH_EAR_CLOSE = ear_th
    
    # Optimization for sweep
    mdl.USE_AUTO_CALIB_THRESH = False # Force manual thresholds for this sweep
    mdl.EXPERIMENT_FRAME_SKIP = 2     # Speed up processing (approx 15fps)
    
    try:
        mdl.process_video(video_path, output_dir, headless=True)
        print(json.dumps({"status": "ok", "video": Path(video_path).stem}))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    run_worker()
