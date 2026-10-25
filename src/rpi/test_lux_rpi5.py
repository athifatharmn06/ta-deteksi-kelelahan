import cv2
import mediapipe as mp
import subprocess
import time
import numpy as np
import sys
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(landmarks, w, h):
    def get_pt(idx): return (landmarks[idx].x * w, landmarks[idx].y * h)
    l_h = euclidean_distance(get_pt(33), get_pt(133))
    if l_h == 0: return 0
    l_v1 = euclidean_distance(get_pt(160), get_pt(144))
    l_v2 = euclidean_distance(get_pt(158), get_pt(153))
    l_ear = (l_v1 + l_v2) / (2.0 * l_h)
    
    r_h = euclidean_distance(get_pt(362), get_pt(263))
    if r_h == 0: return 0
    r_v1 = euclidean_distance(get_pt(385), get_pt(380))
    r_v2 = euclidean_distance(get_pt(387), get_pt(373))
    r_ear = (r_v1 + r_v2) / (2.0 * r_h)
    return (l_ear + r_ear) / 2.0

def calculate_mar(landmarks, w, h):
    def get_pt(idx): return (landmarks[idx].x * w, landmarks[idx].y * h)
    h_dist = euclidean_distance(get_pt(61), get_pt(291))
    if h_dist == 0: return 0
    v_dist1 = euclidean_distance(get_pt(82), get_pt(87))
    v_dist2 = euclidean_distance(get_pt(13), get_pt(14))
    v_dist3 = euclidean_distance(get_pt(312), get_pt(317))
    return (v_dist1 + v_dist2 + v_dist3) / (2.0 * h_dist)

# ============================================
# KAMERA RASPBERRY PI 5 (rpicam-vid wrapper)
# ============================================
class RPiCameraFeed:
    def __init__(self, tuning_file="/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json", width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.use_rpicam = True
        self.process = None
        self.cap = None
        
        try:
            print(f"[INFO] Inisialisasi rpicam-vid streaming dengan tuning: {tuning_file}")
            cmd = [
                "rpicam-vid", "-t", "0", "--tuning-file", tuning_file,
                "--width", str(width), "--height", str(height),
                "--framerate", str(fps), "--codec", "yuv420",
                "--inline", "--nopreview", "-o", "-"
            ]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            self.frame_size = int(width * height * 1.5)
            time.sleep(0.5)
            if self.process.poll() is not None: raise Exception("rpicam-vid command exited immediately.")
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan rpicam-vid: {e}. Fallback ke cv2.VideoCapture(0)")
            self.use_rpicam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def isOpened(self):
        return (self.process is not None and self.process.poll() is None) if self.use_rpicam else (self.cap is not None and self.cap.isOpened())

    def read(self):
        if self.use_rpicam:
            raw_data = self.process.stdout.read(self.frame_size)
            if not raw_data or len(raw_data) != self.frame_size: return False, None
            yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        else: return self.cap.read()

    def release(self):
        if self.use_rpicam and self.process:
            self.process.terminate(); self.process.wait(timeout=2)
        elif self.cap: self.cap.release()

def main():
    print("[INFO] Memulai MediaPipe Face Mesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    tuning_file = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
    cap = RPiCameraFeed(tuning_file=tuning_file, width=1280, height=720)

    if not cap.isOpened():
        print("[FATAL] Kamera RPi IR tidak terdeteksi.")
        sys.exit()

    flip_mode = 0
    print("[INFO] Tekan 'q' untuk keluar, tekan 'f' untuk memutar kamera (flip vertikal).")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[FATAL] Gagal membaca frame.")
            break
            
        frame = cv2.flip(frame, flip_mode)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        h, w, _ = frame.shape
        
        status_text = "FAILED (Fitur Hilang)"
        color = (0, 0, 255) # Merah - Gagal Ekstraksi
        ear_val = 0.0
        mar_val = 0.0

        if results.multi_face_landmarks:
            status_text = "TRACKING (Fitur OK)"
            color = (0, 255, 0) # Hijau - Ekstraksi Berjalan
            
            face_landmarks = results.multi_face_landmarks[0]
            
            ear_val = calculate_ear(face_landmarks.landmark, w, h)
            mar_val = calculate_mar(face_landmarks.landmark, w, h)
            
            # Gambar wireframe agar peneliti bisa lihat apakah landmark distorsi atau tidak
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=frame, landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        # Threshold statis untuk mendemonstrasikan Halusinasi
        ear_state = "TERTUTUP" if (ear_val > 0 and ear_val < 0.22) else "TERBUKA"
        mar_state = "MENGUAP" if mar_val > 0.20 else "TERTUTUP"

        # Latar belakang hitam pekat
        cv2.rectangle(frame, (10, 10), (1050, 220), (0, 0, 0), -1)
        
        cv2.putText(frame, f"FaceMesh: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, f"FaceMesh: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
        
        # Draw EAR State & Value (Untuk mendeteksi halusinasi mata mendadak terbuka)
        ear_color = (0, 255, 0) if ear_state == "TERTUTUP" else (0, 165, 255) # Hijau saat mata benar merem
        txt_ear = f"Mata: {ear_state} (EAR: {ear_val:.3f})"
        cv2.putText(frame, txt_ear, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, txt_ear, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, ear_color, 2, cv2.LINE_AA)
        
        # Draw MAR State & Value (Untuk mendeteksi halusinasi mulut mendadak menutup/terbuka)
        mar_color = (0, 255, 0) if mar_state == "MENGUAP" else (0, 165, 255)
        txt_mar = f"Mulut: {mar_state} (MAR: {mar_val:.3f})"
        cv2.putText(frame, txt_mar, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, txt_mar, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.5, mar_color, 2, cv2.LINE_AA)

        cv2.putText(frame, "RPi 5 + IR Camera Noir", (860, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "Evaluasi Ekstraksi FaceMesh (Lux)", (860, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow("Pengujian 4: Point of Failure FaceMesh (RPi 5 IR Noir)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_mode = -1 if flip_mode == 0 else 0
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
