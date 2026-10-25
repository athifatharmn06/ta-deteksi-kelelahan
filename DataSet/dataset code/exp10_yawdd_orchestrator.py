import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Re-implementing core logic from v5 for the orchestrator to ensure it's standalone and parallelizable
# We focus on the core metrics: EAR, MAR, Head Pose (Pitch/Nodding), and Drowsiness Detection

def calculate_ear(eye_landmarks):
    # Standard EAR formula
    try:
        p1, p2, p3, p4, p5, p6 = eye_landmarks
        dist_v1 = np.linalg.norm(p2 - p6)
        dist_v2 = np.linalg.norm(p3 - p5)
        dist_h = np.linalg.norm(p1 - p4)
        return (dist_v1 + dist_v2) / (2.0 * dist_h)
    except:
        return 0.3 # Default stable value

def calculate_mar(lip_landmarks):
    # Standard MAR formula
    try:
        p1, p2, p3, p4, p5, p6 = lip_landmarks
        dist_v = np.linalg.norm(p2 - p6)
        dist_h = np.linalg.norm(p1 - p4)
        return dist_v / dist_h
    except:
        return 0.2

class HeadPoseEstimator:
    def __init__(self):
        self.pitch_history = []
        self.calibrated_pitch = None
        
    def calculate_pitch(self, face_landmarks, img_w, img_h):
        # Using the geometric projection method from v5
        nose_tip = np.array([face_landmarks.landmark[1].x * img_w, face_landmarks.landmark[1].y * img_h])
        forehead = np.array([face_landmarks.landmark[10].x * img_w, face_landmarks.landmark[10].y * img_h])
        chin = np.array([face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h])
        
        v_face = forehead - chin
        v_nose = nose_tip - chin
        
        mag_face = np.linalg.norm(v_face)
        if mag_face == 0: return 0
        
        projection = np.dot(v_nose, v_face) / mag_face
        pitch = (projection / mag_face) * 100
        return pitch

def process_single_video(video_path, subject_id, gender, ethnicity, eyewear, view_type):
    """Worker function for a single YawDD video."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    pose_estimator = HeadPoseEstimator()
    frames_count = 0
    drowsy_events = 0
    yawn_events = 0
    nod_events = 0
    processed_frames = 0
    
    # Simple state machines for events
    eye_closed_frames = 0
    mouth_open_frames = 0
    nodding_frames = 0
    
    # Thresholds (Optimized from Exp 2)
    EAR_THRESHOLD = 0.16
    MAR_THRESHOLD = 0.26
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed_frames += 1
        # Process every 2nd frame for speed (YawDD is high FPS)
        if processed_frames % 2 != 0: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # EAR (Left: 362, 385, 386, 263, 374, 380 | Right: 33, 160, 158, 133, 153, 144)
            left_eye_indices = [362, 385, 386, 263, 374, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            
            l_eye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in left_eye_indices])
            r_eye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in right_eye_indices])
            
            ear = (calculate_ear(l_eye) + calculate_ear(r_eye)) / 2.0
            
            # MAR (Outer lip: 61, 291, 0, 17, 39, 181 ... using internal points 13, 14, 78, 308)
            mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178] # Internal
            mouth = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in mouth_indices])
            # Simplified MAR for speed
            mar = calculate_mar(mouth[:6]) 
            
            # Pitch
            pitch = pose_estimator.calculate_pitch(landmarks, w, h)
            
            # Drowsiness (EAR)
            if ear < EAR_THRESHOLD:
                eye_closed_frames += 2 # because skip
                if eye_closed_frames >= fps: # 1 second closed
                    drowsy_events += 1
                    eye_closed_frames = 0
            else:
                eye_closed_frames = 0
                
            # Yawn (MAR)
            if mar > MAR_THRESHOLD:
                mouth_open_frames += 2
                if mouth_open_frames >= (fps * 1.5): # 1.5 second yawn
                    yawn_events += 1
                    mouth_open_frames = 0
            else:
                mouth_open_frames = 0
                
            # Nodding (Pitch change - simplified)
            if pose_estimator.calibrated_pitch is None:
                if processed_frames < fps * 2:
                    pose_estimator.pitch_history.append(pitch)
                    if len(pose_estimator.pitch_history) > 30:
                        pose_estimator.calibrated_pitch = np.mean(pose_estimator.pitch_history)
            else:
                diff = pitch - pose_estimator.calibrated_pitch
                if diff > 10: # Threshold for nodding
                    nodding_frames += 2
                    if nodding_frames >= (fps / 2):
                        nod_events += 1
                        nodding_frames = 0
                else:
                    nodding_frames = 0

    cap.release()
    face_mesh.close()
    
    return {
        "video": Path(video_path).name,
        "subject": subject_id,
        "gender": gender,
        "ethnicity": ethnicity,
        "eyewear": eyewear,
        "view": view_type,
        "drowsy_count": drowsy_events,
        "yawn_count": yawn_events,
        "nod_count": nod_events,
        "frames": processed_frames
    }

def main():
    root_dir = Path(r"C:\TA Deteksi Kelelahan\DataSet\YawDD")
    demo_file = root_dir / "Demografi_yawdd.json"
    results_dir = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(demo_file, 'r') as f:
        demo_data = json.load(f)
        
    tasks = []
    
    # 1. Dash Dataset
    dash_map = {item['id']: item for item in demo_data['dash_dataset']}
    dash_dir = root_dir / "Dash" / "Dash"
    for sex in ["Female", "Male"]:
        d = dash_dir / sex
        for v in d.glob("*.avi"):
            sid_str = v.name.split('-')[0]
            sid = f"{sex[0]}{sid_str}"
            info = dash_map.get(sid, {"gender": sex, "ethnicity": "Unknown", "glasses": "Unknown"})
            tasks.append((v, sid, info['gender'], info['ethnicity'], info['glasses'], "Dashboard"))

    # 2. Mirror Dataset
    mirror_map = {item['id']: item for item in demo_data['mirror_dataset']}
    mirror_root = root_dir / "Mirror" / "Mirror"
    
    # Use list of subdirs for speed
    female_mirror = mirror_root / "Female_mirror"
    male_mirror = mirror_root / "Male_mirror Avi Videos"
    
    for d, tag in [(female_mirror, "Female"), (male_mirror, "Male")]:
        for v in d.glob("*.avi"):
            sid_str = v.name.split('-')[0]
            sid = f"{tag[0]}{sid_str}"
            info = mirror_map.get(sid, {"gender": tag, "ethnicity": "Unknown", "glasses": "Unknown"})
            tasks.append((v, sid, info['gender'], info['ethnicity'], info['glasses'], "Mirror"))

    print(f"[START] Processing {len(tasks)} videos using 4 workers...")
    
    results = []
    # Using 4 workers to balance CPU and MediaPipe overhead
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_video = {executor.submit(process_single_video, *t): t[0] for t in tasks}
        
        finished = 0
        for future in as_completed(future_to_video):
            res = future.result()
            if res:
                results.append(res)
            finished += 1
            if finished % 10 == 0:
                print(f"  Progress: {finished}/{len(tasks)} videos done...")

    # Save Results
    df_results = pd.DataFrame(results) if 'pd' in locals() else None # We can use CSV directly
    csv_path = results_dir / "exp10_yawdd_raw_results.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    # Generate Summary
    summary = {
        "exp": "Exp 10: YawDD Cross-Dataset Evaluation",
        "total_videos": len(results),
        "by_view": {},
        "by_gender": {}
    }
    
    # Quick aggregation
    for v_type in ["Dashboard", "Mirror"]:
        sub = [r for r in results if r['view'] == v_type]
        if sub:
            summary['by_view'][v_type] = {
                "count": len(sub),
                "avg_drowsy": round(sum(r['drowsy_count'] for r in sub) / len(sub), 2),
                "avg_yawn": round(sum(r['yawn_count'] for r in sub) / len(sub), 2)
            }
            
    json_path = results_dir / "exp10_yawdd_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n[DONE] Cross-Dataset Evaluation Complete.")
    print(f"  Results saved to {results_dir}")

if __name__ == "__main__":
    main()
