"""
ablation_worker.py — Process one video with a specific ablation config.
Usage: python ablation_worker.py <video_path> <config> <output_dir>
Configs: C1, C2, C3, C4
Writes JSON result to a .result file in the output dir.
"""
import sys
import os
import json
import csv
import io
from pathlib import Path

# Parse args BEFORE importing v5
if len(sys.argv) < 4:
    sys.exit(1)

video_path = sys.argv[1]
config     = sys.argv[2].upper()
output_dir = sys.argv[3]

# Suppress pygame welcome message and fix Unicode
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Redirect stdout to suppress v5 noise — we'll write result to a file instead
# Must use utf-8 because v5 prints unicode arrows (→)
real_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w', encoding='utf-8')

# Import v5 — this triggers MediaPipe init (prints to stdout, now silenced)
import v5_own_detection_model as mdl

# ============================================================
# SET FLAGS BASED ON CONFIG
# ============================================================
CONFIGS = {
    'C1': {'auto_calib': False, 'peak_gate': False, 'osc_guard': False, 'mar_thresh': 0.30},
    'C2': {'auto_calib': True,  'peak_gate': False, 'osc_guard': False, 'mar_thresh': None},
    'C3': {'auto_calib': True,  'peak_gate': True,  'osc_guard': False, 'mar_thresh': None},
    'C4': {'auto_calib': True,  'peak_gate': True,  'osc_guard': True,  'mar_thresh': None},
}

if config not in CONFIGS:
    sys.stdout = real_stdout
    print(json.dumps({"error": f"Unknown config: {config}"}))
    sys.exit(1)

cfg = CONFIGS[config]
mdl.USE_AUTO_CALIB_THRESH = cfg['auto_calib']
mdl.USE_PEAK_GATE         = cfg['peak_gate']
mdl.USE_OSCILLATION_GUARD = cfg['osc_guard']

# For C1: set fixed MAR threshold
if cfg['mar_thresh'] is not None:
    mdl.THRESH_MAR_YAWN = cfg['mar_thresh']

# ============================================================
# RUN
# ============================================================
video_stem = Path(video_path).stem
config_outdir = os.path.join(output_dir, config, video_stem)
os.makedirs(config_outdir, exist_ok=True)

try:
    mdl.process_video(video_path, config_outdir, headless=True)
except Exception as e:
    result = {"error": str(e), "video": os.path.basename(video_path), "config": config}
    result_file = Path(config_outdir) / "_ablation_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f)
    sys.exit(1)

# ============================================================
# READ OUTPUT CSV & WRITE RESULT FILE
# ============================================================
yawn_csv = Path(config_outdir) / f"{video_stem}_yawning.csv"
yawn_events = []
if yawn_csv.exists():
    with open(yawn_csv) as f:
        for row in csv.DictReader(f):
            yawn_events.append({
                'frame_start': int(row['frame_start']),
                'frame_end':   int(row['frame_end']),
                'duration':    float(row['duration_sec']),
            })

result = {
    "video": os.path.basename(video_path),
    "config": config,
    "yawn_count": len(yawn_events),
    "yawn_events": yawn_events,
}

# Write result to file (not stdout — v5 pollutes stdout)
result_file = Path(config_outdir) / "_ablation_result.json"
with open(result_file, 'w') as f:
    json.dump(result, f)
