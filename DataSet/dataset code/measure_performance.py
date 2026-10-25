import os
import sys
import traceback
from pathlib import Path

# Redirect stdout and stderr to a normal text file
f_out = open('perf_logs_clean.txt', 'w', encoding='utf-8')
sys.stdout = f_out
sys.stderr = f_out

try:
    import v5_own_detection_model as mdl
    
    if __name__ == "__main__":
        mdl.USE_OSCILLATION_GUARD = True
        mdl.USE_PEAK_GATE = True
        mdl.USE_PERCLOS_WARMUP = True

        current_dir = Path(os.path.abspath(os.path.dirname(__file__)))
        dataset_base = current_dir.parent
        ds_folder = dataset_base / "DMD Dataset 1"
        print(f"Checking folder: {ds_folder}")
        vids = list(ds_folder.glob("*_ir_face.mp4"))
        if not vids:
             vids = list(ds_folder.glob("*.mp4"))
        
        print(f"Found videos: {[v.name for v in vids]}")
        
        if not vids:
            print("No videos found!")
            sys.exit(1)
             
        video_path = str(vids[0])
        out_dir = current_dir / "perf_output"
        print(f"Mengukur performa di video: {video_path}")
        mdl.process_video(video_path, str(out_dir), headless=True)
except Exception as e:
    print("CRASHED:", e)
    traceback.print_exc()
finally:
    f_out.flush()
    f_out.close()
