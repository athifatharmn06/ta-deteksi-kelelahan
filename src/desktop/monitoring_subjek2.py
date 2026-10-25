import cv2
import mediapipe as mp
import time
import math
import numpy as np
import sys
from pathlib import Path

RPI_SRC_DIR = Path(__file__).resolve().parents[1] / "rpi"
sys.path.insert(0, str(RPI_SRC_DIR))

import v5_rpi5_drowsiness_detection as detect

# CONFIG
VIDEO_PATH = r"c:\TA Deteksi Kelelahan\Laporan Progress\Pengujian 1 dan 2\video\Subjek2_Gelap_0Lux.avi"
START_SEC = 0
END_SEC = 120 # Cek 2 menit pertama saja agar cepat

def run_monitoring():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Gagal membuka video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    
    # Init MediaPipe
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)

    # Calibration Values (Based on Subjek 2 profile)
    be, bm, bp = 0.28, 0.08, 0.0
    det = detect.DrowsinessDetector(fps=int(fps), baseline_ear=be, baseline_mouth=bm, baseline_pitch=bp)
    
    print(f"Memulai Monitoring Subjek 2 Gelap...")
    cap.set(cv2.CAP_PROP_POS_MSEC, START_SEC * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        curr_sec = curr_ms / 1000.0
        if curr_sec > END_SEC: break

        # Process
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        
        h, w = frame.shape[:2]
        ear, mar, pitch = be, bm, bp # Default
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            def gp(i): return (int(lm[i].x*w), int(lm[i].y*h))
            
            # Simple EAR/MAR for visualization
            def d(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            p_e = [gp(i) for i in [33,160,158,133,153,144]]
            ear = (d(p_e[1],p_e[5]) + d(p_e[2],p_e[4])) / (2 * d(p_e[0],p_e[3]))
            
            p_m = [gp(i) for i in [13, 14, 61, 291]]
            mar = d(p_m[0], p_m[1]) / d(p_m[2], p_m[3])
            
            # Update Detector
            r = det.update(ear, mar, bp, int(fps))
            
            # DRAW HUD
            cv2.rectangle(frame, (10, 10), (250, 150), (0,0,0), -1)
            cv2.putText(frame, f"TIME: {curr_sec:.1f}s", (20, 40), 1, 1.2, (255,255,255), 2)
            cv2.putText(frame, f"EAR: {ear:.3f}", (20, 70), 1, 1.2, (0,255,0) if ear > 0.2 else (0,0,255), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (20, 100), 1, 1.2, (0,255,255) if mar > 0.4 else (255,255,255), 2)
            
            status = "AMAN"
            color = (0, 255, 0)
            if r['microsleep']: status, color = "MICROSLEEP!", (0,0,255)
            elif r['yawning']: status, color = "MENGUAP!", (0,255,255)
            elif r['eye_closed']: status, color = "MATA TERTUTUP", (0,165,255)
            
            cv2.putText(frame, f"STATUS: {status}", (20, 135), 1, 1.5, color, 3)

        cv2.imshow("MONITORING SUBJEK 2 GELAP", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monitoring()
