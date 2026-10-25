import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import json
from pathlib import Path
from collections import deque

# ============================================
# GEOMETRIC UTILITIES (Identical to v5 model)
# ============================================

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_point(landmark, index, width, height):
    return (int(landmark[index].x * width), int(landmark[index].y * height))

def calculate_head_pose_improved(landmarks, width, height):
    """Pitch calculation using dot-product projection of nose tip."""
    try:
        forehead = get_landmark_point(landmarks, 10,  width, height)
        chin     = get_landmark_point(landmarks, 152, width, height)
        nose_tip = get_landmark_point(landmarks, 4,   width, height)

        ax = chin[0] - forehead[0]
        ay = chin[1] - forehead[1]
        face_len_sq = ax * ax + ay * ay

        if face_len_sq < 1.0:
            return 0.0

        nx = nose_tip[0] - forehead[0]
        ny = nose_tip[1] - forehead[1]

        # Projection onto facial axis
        t = (nx * ax + ny * ay) / face_len_sq
        # t=0.45 is neutral. Smaller t = looking up, larger t = looking down.
        pitch = (0.45 - t) * 120.0
        return max(-45.0, min(45.0, pitch))
    except:
        return 0.0

class PitchCalibrator:
    """Finds the most stable window from the collected samples."""
    def __init__(self, fps=30, collect_seconds=5.0, window_seconds=1.0):
        self.fps = fps
        self.collect_frames = int(collect_seconds * fps)
        self.window_frames = max(1, int(window_seconds * fps))
        self.pitch_samples = []
        self.baseline_pitch = 0.0
        self.stable_std = 999.0
        self.done = False

    def add_sample(self, pitch_raw):
        if self.done: return True
        self.pitch_samples.append(pitch_raw)
        if len(self.pitch_samples) >= self.collect_frames:
            self._compute()
            return True
        return False

    def _compute(self):
        vals = np.array(self.pitch_samples)
        n, wf = len(vals), self.window_frames
        best_std, best_start = float('inf'), 0
        for i in range(n - wf + 1):
            s = float(np.std(vals[i : i + wf]))
            if s < best_std:
                best_std, best_start = s, i
        best_window = vals[best_start : best_start + wf]
        self.baseline_pitch = float(np.median(best_window))
        self.stable_std = best_std
        self.done = True

# ============================================
# EXPERIMENT ORCHESTRATOR
# ============================================

def process_single_video(video_path):
    print(f"\n[CALIBRATING] {video_path.name}...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    # Process only first 7 seconds to be safe (5s for calib)
    max_frames = int(fps * 7)
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        calibrator = PitchCalibrator(fps=fps)
        frame_idx = 0
        
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            if res.multi_face_landmarks:
                for landmarks in res.multi_face_landmarks:
                    pitch = calculate_head_pose_improved(landmarks.landmark, w, h)
                    calibrator.add_sample(pitch)
                    if calibrator.done:
                        break
            if calibrator.done:
                break
                
    cap.release()
    if calibrator.done:
        return {
            "video": video_path.name,
            "baseline_pitch": round(calibrator.baseline_pitch, 3),
            "stable_std": round(calibrator.stable_std, 4),
            "samples": len(calibrator.pitch_samples)
        }
    return None

def main():
    # Use absolute base path to be safe
    dataset_dir = Path(r"c:\TA Deteksi Kelelahan\DataSet")
    results_dir = Path(r"c:\TA Deteksi Kelelahan\Jurnal JNTETI\Results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"[DEBUG] Searching in: {dataset_dir}")
    
    # Discovery (DMD Dataset 1-15)
    video_list = []
    for i in range(1, 16):
        d = dataset_dir / f"DMD Dataset {i}"
        if not d.exists():
            print(f"[DEBUG] Directory not found: {d}")
            continue
        
        found_in_dir = list(d.glob("*_ir_face.mp4"))
        print(f"[DEBUG] Folder {d.name}: found {len(found_in_dir)} videos")
        video_list.extend(found_in_dir)
            
    print(f"[FOUND] {len(video_list)} videos for Exp 6")
    
    if not video_list:
        print("[ERROR] No videos found! Check data structure.")
        return
    
    results = []
    for v in video_list:
        res = process_single_video(v)
        if res:
            print(f"  Result: Baseline={res['baseline_pitch']}, Std={res['stable_std']}")
            results.append(res)
            
    # Save CSV
    csv_path = results_dir / "exp6_pitch_calibration.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["video", "baseline_pitch", "stable_std", "samples"])
        w.writeheader()
        w.writerows(results)
        
    # Aggregate Stats
    pitches = [r["baseline_pitch"] for r in results]
    stds = [r["stable_std"] for r in results]
    
    summary = {
        "exp": "Exp 6: Pitch Calibration Stability",
        "total_subjects": len(results),
        "pitch": {
            "mean": round(float(np.mean(pitches)), 3),
            "std": round(float(np.std(pitches)), 3),
            "min": round(float(np.min(pitches)), 3),
            "max": round(float(np.max(pitches)), 3)
        },
        "stability": {
            "mean_std": round(float(np.mean(stds)), 4),
            "max_std": round(float(np.max(stds)), 4),
            "subjects_stable_count": sum(1 for s in stds if s < 0.5)
        }
    }
    
    json_path = results_dir / "exp6_pitch_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[DONE] Exp 6 complete. Saved to:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    print(f"\nSummary: Mean Pitch={summary['pitch']['mean']} deg, Mean Std={summary['stability']['mean_std']}")

if __name__ == "__main__":
    main()
