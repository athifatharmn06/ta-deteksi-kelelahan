import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import math
import subprocess

# ============================================
# KONFIGURASI SISTEM
# ============================================
# Bobot deteksi
W_EYE = 0.45        # Bobot mata
W_MOUTH = 0.25      # Bobot mulut
W_HEAD = 0.30       # Bobot kepala

# ABLATION CONFIG FLAGS
USE_OSCILLATION_GUARD = True   # Ganti False untuk C1, C2
USE_PEAK_GATE         = True   # Ganti False untuk C1
USE_PERCLOS_WARMUP    = True   # Ganti False untuk C1, C2, C3

# Threshold default
THRESH_EAR_CLOSE = 0.18      # EAR untuk mata tertutup
THRESH_EAR_BLINK = 0.22      # EAR untuk kedipan
THRESH_MAR_YAWN = 0.20       # MAR untuk menguap
THRESH_HEAD_PITCH = 4.0      # Pitch untuk nunduk (derajat)

# Parameter waktu
TIME_MICROSLEEP_LIMIT = 1.5  # Microsleep (1.5 detik)
TIME_YAWN_MIN_DURATION = 1.5  # Menguap minimal 1.5 detik
TIME_BLINK_MIN_INTERVAL = 0.2  # Interval kedip minimum
TIME_CONSECUTIVE_FRAMES = 15  # Frame berturut-turut untuk konfirmasi
TIME_PERCLOS_WINDOW = 60      # Window PERCLOS (60 detik)

# Warna UI
COLOR_SAFE = (0, 255, 0)     # Hijau
COLOR_WARN = (0, 165, 255)   # Oranye
COLOR_DANGER = (0, 0, 255)   # Merah
COLOR_INFO = (200, 200, 200) # Abu-abu terang
COLOR_DEBUG = (255, 255, 0)  # Kuning
COLOR_ROI = (0, 255, 255)    # Kuning untuk ROI

# ============================================
# KAMERA RASPBERRY PI 5 (rpicam-vid wrapper)
# ============================================
class RPiCameraFeed:
    """Wrapper untuk membaca YUV420 raw dari stdout rpicam-vid untuk Raspberry Pi 5"""
    def __init__(self, tuning_file="/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json", width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.use_rpicam = True
        self.process = None
        self.cap = None
        
        try:
            print(f"[INFO] Inisialisasi rpicam-vid streaming dengan tuning: {tuning_file}")
            cmd = [
                "rpicam-vid",
                "-t", "0",
                "--tuning-file", tuning_file,
                "--width", str(width),
                "--height", str(height),
                "--framerate", str(fps),
                "--codec", "yuv420",
                "--inline",
                "--nopreview",
                "-o", "-"
            ]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            self.frame_size = int(width * height * 1.5)
            
            # Cek cepat jika mati di awal
            time.sleep(0.5)
            if self.process.poll() is not None:
                raise Exception("rpicam-vid command exited immediately.")
                
        except Exception as e:
            print(f"[WARNING] Gagal menjalankan rpicam-vid: {e}. Fallback ke cv2.VideoCapture(0)")
            self.use_rpicam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def isOpened(self):
        if self.use_rpicam:
            return self.process is not None and self.process.poll() is None
        else:
            return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.use_rpicam:
            raw_data = self.process.stdout.read(self.frame_size)
            if not raw_data or len(raw_data) != self.frame_size:
                return False, None
            # Dekode YUV420 ke BGR
            yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        else:
            return self.cap.read()

    def release(self):
        if self.use_rpicam and self.process:
            self.process.terminate()
            self.process.wait(timeout=2)
        elif self.cap:
            self.cap.release()

# ============================================
# FUNGSI UTILITAS GEOMETRI
# ============================================
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_point(landmark, index, width, height):
    return (int(landmark[index].x * width), int(landmark[index].y * height))

def calculate_ear_single_eye(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def calculate_ear(landmarks, width, height):
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    left_eye_points = [get_landmark_point(landmarks, i, width, height) for i in left_eye_indices]
    
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_points = [get_landmark_point(landmarks, i, width, height) for i in right_eye_indices]
    
    left_ear = calculate_ear_single_eye(left_eye_points)
    right_ear = calculate_ear_single_eye(right_eye_points)
    
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, left_ear, right_ear

def calculate_mar(landmarks, width, height):
    left_corner  = get_landmark_point(landmarks, 61,  width, height)
    right_corner = get_landmark_point(landmarks, 291, width, height)

    upper_left  = get_landmark_point(landmarks, 82,  width, height)
    lower_left  = get_landmark_point(landmarks, 87,  width, height)

    upper_mid   = get_landmark_point(landmarks, 13,  width, height)
    lower_mid   = get_landmark_point(landmarks, 14,  width, height)

    upper_right = get_landmark_point(landmarks, 312, width, height)
    lower_right = get_landmark_point(landmarks, 317, width, height)

    A = euclidean_distance(upper_left,  lower_left)
    B = euclidean_distance(upper_mid,   lower_mid)
    C = euclidean_distance(upper_right, lower_right)
    D = euclidean_distance(left_corner, right_corner)

    if D == 0: return 0.0
    return (A + B + C) / (2.0 * D)

def calculate_head_pose_improved(landmarks, width, height):
    try:
        forehead = get_landmark_point(landmarks, 10,  width, height)
        chin     = get_landmark_point(landmarks, 152, width, height)
        nose_tip = get_landmark_point(landmarks, 4,   width, height)

        ax = chin[0] - forehead[0]
        ay = chin[1] - forehead[1]
        face_len_sq = ax * ax + ay * ay

        if face_len_sq < 1.0: return 0.0

        nx = nose_tip[0] - forehead[0]
        ny = nose_tip[1] - forehead[1]

        t = (nx * ax + ny * ay) / face_len_sq
        pitch = (0.45 - t) * 120.0
        return max(-45.0, min(45.0, pitch))
    except Exception as e:
        return 0.0

# ============================================
# PERCLOS CALCULATOR
# ============================================
class PERCLOSCalculator:
    def __init__(self, window_seconds=60, fps=30):
        self.window_seconds = window_seconds
        self.fps = fps
        self.window_size = int(window_seconds * fps)

        self.warmup_seconds = 30
        self.warmup_frames  = int(self.warmup_seconds * fps)
        
        self.eye_states = deque(maxlen=self.window_size)
        
        self.total_frames = 0
        self.closed_frames = 0
        self.last_perclos = 0.0
        
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0
        self.last_eye_state = False
        
    def update(self, ear, threshold):
        is_closed = ear < threshold
        if self.last_eye_state == True and is_closed == False:
            self.blink_count += 1
        self.last_eye_state = is_closed
        
        self.eye_states.append(is_closed)
        self.total_frames = len(self.eye_states)
        self.closed_frames = sum(1 for state in self.eye_states if state)
        
        if (not USE_PERCLOS_WARMUP) or (self.total_frames >= self.warmup_frames):
            self.last_perclos = (self.closed_frames / self.total_frames) * 100.0
        else:
            self.last_perclos = 0.0
        
        current_time = time.time()
        time_interval = current_time - self.last_blink_time
        
        if time_interval >= 60:
            self.blink_rate = (self.blink_count / time_interval) * 60
            self.blink_count = 0
            self.last_blink_time = current_time
        
        return {
            'perclos': self.last_perclos,
            'blink_rate': self.blink_rate,
            'eye_closed': is_closed,
            'window_frames': self.total_frames,
            'closed_frames': self.closed_frames,
            'blink_count': self.blink_count
        }

# ============================================
# DROWSINESS DETECTOR
# ============================================
class DrowsinessDetector:
    def __init__(self, fps=30, baseline_pitch=0.0):
        self.ear_buffer = deque(maxlen=5)
        self.mar_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=5)
        
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.head_down_frames = 0
        
        self.microsleep_start = None
        self.blink_count = 0
        self.yawn_count = 0
        self.total_blinks = 0
        self.total_yawns = 0
        
        self._blink_in_progress = False
        self._blink_start_frame = 0
        self._blink_frame_counter = 0
        self.BLINK_MIN_FRAMES = 2
        self.BLINK_MAX_FRAMES = 15
        self.valid_blink_count = 0
        
        self.normal_ear = None
        self.normal_mar = None
        self.last_mar = 0.0
        self.mar_increasing = False

        self._yawn_state      = 'IDLE'
        self.yawn_hold_frames = 0
        self.yawn_peak_mar    = 0.0
        self.last_yawn_time   = 0.0
        self.yawn_cooldown    = 5.0

        self._mar_above_prev  = False
        self._mar_cross_times = deque()
        self._osc_window      = 4.0
        self._osc_max         = 3
        
        self.perclos_calc = PERCLOSCalculator(window_seconds=TIME_PERCLOS_WINDOW, fps=fps)
        self.is_yawning = False
        self.baseline_pitch = baseline_pitch
        self.pitch_threshold = THRESH_HEAD_PITCH
        
    def smooth_value(self, buffer, new_value):
        buffer.append(new_value)
        return sum(buffer) / len(buffer)
    
    def update(self, ear, mar, pitch, fps=30):
        ear_smooth = self.smooth_value(self.ear_buffer, ear)
        mar_smooth = self.smooth_value(self.mar_buffer, mar)
        pitch_smooth = self.smooth_value(self.pitch_buffer, pitch)
        
        if self.normal_ear is None: self.normal_ear = ear_smooth
        if self.normal_mar is None: self.normal_mar = mar_smooth
        
        if mar_smooth > self.last_mar * 1.05:
            self.mar_increasing = True
        elif mar_smooth < self.last_mar * 0.95:
            self.mar_increasing = False
        self.last_mar = mar_smooth
        
        perclos_data = self.perclos_calc.update(ear_smooth, THRESH_EAR_CLOSE)
        self.blink_count = perclos_data['blink_count']
        eye_closed = ear_smooth < THRESH_EAR_CLOSE
        
        if eye_closed and not self._blink_in_progress:
            self._blink_in_progress = True
            self._blink_frame_counter = 1
        elif eye_closed and self._blink_in_progress:
            self._blink_frame_counter += 1
        elif not eye_closed and self._blink_in_progress:
            if self.BLINK_MIN_FRAMES <= self._blink_frame_counter <= self.BLINK_MAX_FRAMES:
                self.valid_blink_count += 1
            self._blink_in_progress = False
            self._blink_frame_counter = 0
        
        pitch_deviation = pitch_smooth - self.baseline_pitch
        head_down = pitch_deviation < -self.pitch_threshold and pitch_smooth < -3.0
        head_up = pitch_deviation > self.pitch_threshold and pitch_smooth > 3.0
        
        mar_threshold = THRESH_MAR_YAWN
        peak_gate_abs = max(mar_threshold * 1.6, 0.50)
        mar_above_now = mar_smooth > mar_threshold
        current_time_now = time.time()

        if mar_above_now and not self._mar_above_prev:
            self._mar_cross_times.append(current_time_now)
        while self._mar_cross_times and (current_time_now - self._mar_cross_times[0]) > self._osc_window:
            self._mar_cross_times.popleft()
            
        if USE_OSCILLATION_GUARD:
            is_talking = len(self._mar_cross_times) > self._osc_max
        else:
            is_talking = False
            
        self._mar_above_prev = mar_above_now

        if self._yawn_state == 'IDLE':
            if mar_above_now and not is_talking:
                self._yawn_state      = 'BUILDING'
                self.yawn_hold_frames = 1
                self.yawn_peak_mar    = mar_smooth

        elif self._yawn_state == 'BUILDING':
            if not mar_above_now or is_talking:
                self._yawn_state      = 'IDLE'
                self.yawn_hold_frames = 0
                self.yawn_peak_mar    = 0.0
            else:
                self.yawn_hold_frames += 1
                if mar_smooth > self.yawn_peak_mar:
                    self.yawn_peak_mar = mar_smooth
                min_hold = int(TIME_YAWN_MIN_DURATION * fps)
                peak_condition = (self.yawn_peak_mar >= peak_gate_abs) if USE_PEAK_GATE else True
                if self.yawn_hold_frames >= min_hold and peak_condition:
                    self._yawn_state  = 'CONFIRMED'
                    self.is_yawning   = True
                    self.yawn_count  += 1
                    self.total_yawns += 1
                    self.last_yawn_time = current_time_now

        elif self._yawn_state == 'CONFIRMED':
            if not mar_above_now:
                self._yawn_state      = 'COOLDOWN'
                self.is_yawning       = False
                self.yawn_hold_frames = 0
                self.yawn_peak_mar    = 0.0

        elif self._yawn_state == 'COOLDOWN':
            self.is_yawning = False
            if (current_time_now - self.last_yawn_time) >= self.yawn_cooldown:
                self._yawn_state = 'IDLE'

        yawn_detected = (self._yawn_state == 'CONFIRMED')
        
        if eye_closed and not self.is_yawning and mar_smooth < mar_threshold * 0.7:
            self.eye_closed_frames = min(self.eye_closed_frames + 1, TIME_CONSECUTIVE_FRAMES * 2)
        else:
            self.eye_closed_frames = max(self.eye_closed_frames - 2, 0)
        
        microsleep_detected = False
        if (self.eye_closed_frames >= TIME_CONSECUTIVE_FRAMES and 
            not self.is_yawning and mar_smooth < mar_threshold * 0.6):
            if self.microsleep_start is None:
                self.microsleep_start = time.time()
            elif time.time() - self.microsleep_start > TIME_MICROSLEEP_LIMIT:
                microsleep_detected = True
        else:
            self.microsleep_start = None
        
        if head_down:
            self.head_down_frames = min(self.head_down_frames + 1, TIME_CONSECUTIVE_FRAMES)
        else:
            self.head_down_frames = max(self.head_down_frames - 1, 0)
        
        head_down_detected = self.head_down_frames >= TIME_CONSECUTIVE_FRAMES
        
        if (not yawn_detected and not eye_closed and 
            mar_smooth < mar_threshold * 0.5 and not head_down and not head_up):
            if self.normal_ear is not None: self.normal_ear = self.normal_ear * 0.98 + ear_smooth * 0.02
            if self.normal_mar is not None: self.normal_mar = self.normal_mar * 0.98 + mar_smooth * 0.02
        
        score = 0
        if microsleep_detected:
            score += W_EYE * 100
        elif self.eye_closed_frames >= TIME_CONSECUTIVE_FRAMES and not self.is_yawning:
            score += W_EYE * 60
        elif perclos_data['perclos'] > 20:
            score += W_EYE * (min(perclos_data['perclos'], 40) / 40 * 70)
        
        if yawn_detected:
            score += W_MOUTH * 70
        elif self.yawn_frames > int(TIME_YAWN_MIN_DURATION * fps * 0.3):
            score += W_MOUTH * 20
        
        if head_down_detected:
            score += W_HEAD * 100
        elif self.head_down_frames > TIME_CONSECUTIVE_FRAMES * 0.5:
            score += W_HEAD * 40
        elif head_up:
            score += W_HEAD * 30
        
        score = max(0, min(100, score))
        
        return {
            'ear': ear_smooth,
            'mar': mar_smooth,
            'pitch': pitch_smooth,
            'pitch_deviation': pitch_deviation,
            'perclos': perclos_data['perclos'],
            'blink_rate': perclos_data['blink_rate'],
            'eye_closed': eye_closed,
            'microsleep': microsleep_detected,
            'yawning': yawn_detected,
            'head_down': head_down_detected,
            'head_down_raw': head_down,
            'head_up': head_up,
            'score': score,
            'blink_count': self.blink_count,
            'yawn_count': self.yawn_count,
            'total_yawns': self.total_yawns,
            'valid_blink_count': self.valid_blink_count,
            'window_frames': perclos_data['window_frames'],
            'closed_frames': perclos_data['closed_frames'],
            'normal_ear': self.normal_ear,
            'normal_mar': self.normal_mar,
            'yawn_peak_mar': self.yawn_peak_mar,
            'yawn_state': self._yawn_state,
            'yawn_hold_frames': self.yawn_hold_frames,
            'is_talking': is_talking
        }


# ============================================
# FUNGSI UTAMA (MAIN LOOP & GUI)
# ============================================
def main():
    global THRESH_EAR_CLOSE, THRESH_MAR_YAWN, THRESH_HEAD_PITCH, THRESH_EAR_BLINK

    print("[INFO] Memulai MediaPipe...")
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    print("[INFO] Membuka Kamera...")
    tuning_file = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
    cap = RPiCameraFeed(tuning_file=tuning_file, width=1280, height=720)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[FATAL] Tidak ada kamera yang terdeteksi. Keluar.")
        sys.exit()

    if hasattr(cap, 'width'):
        w = cap.width
        h = cap.height
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolusi: {w}x{h}")
    
    detector = None
    state = "CALIBRATION"
    calibration_data = []
    baseline_values = {}
    
    fps = 30
    frame_count = 0
    start_time = time.time()
    calibration_start_time = None
    auto_calib_start = None
    flip_mode = 0
    
    roi_preview_size = 120
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.55
    FONT_NORMAL = 0.65
    FONT_THICKNESS = 1
    
    ear_history = deque(maxlen=100)
    mar_history = deque(maxlen=100)
    pitch_history = deque(maxlen=100)
    
    def draw_text_with_outline(img, text, pos, font, scale, color, thickness):
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    print("\n[INFO] Sistem siap. Tekan:")
    print("  'c' - Mulai Kalibrasi (3 detik)")
    print("  'q' - Keluar")
    print("  'r' - Reset")
    print("  's' - Simpan Screenshot")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[FATAL] Koneksi ke kamera terputus! (cap.read() = False)")
                break
                
            frame = cv2.flip(frame, flip_mode) # Toggleable flip mode. Default 0 (vertical) for upside-down. Use 'f' to toggle to -1.
            frame_original = frame.copy()
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            roi_frame = None
            roi_coords = None
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_detection = face_detection.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results_detection.detections:
                for detection in results_detection.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w_bbox = int(bboxC.width * iw)
                    h_bbox = int(bboxC.height * ih)
                    
                    padding = 30
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(iw, x + w_bbox + padding)
                    y2 = min(ih, y + h_bbox + padding)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)
                    cv2.putText(frame, "BlazeFace ROI", (x1, y1-10), FONT, FONT_SMALL, COLOR_ROI, FONT_THICKNESS)
                    
                    roi_frame = frame_original[y1:y2, x1:x2]
                    
                    if roi_frame is not None and roi_frame.shape[0] > 0 and roi_frame.shape[1] > 0:
                        roi_h, roi_w, _ = roi_frame.shape
                        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                        roi_rgb.flags.writeable = False
                        results_mesh = face_mesh.process(roi_rgb)
                        roi_rgb.flags.writeable = True
                        
                        if results_mesh.multi_face_landmarks:
                            for face_landmarks in results_mesh.multi_face_landmarks:
                                for landmark in face_landmarks.landmark:
                                    landmark.x = (landmark.x * roi_w + x1) / iw
                                    landmark.y = (landmark.y * roi_h + y1) / ih
                                
                                # Draw some landmarks
                                for i in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                                    cv2.circle(frame, get_landmark_point(face_landmarks.landmark, i, iw, ih), 2, (0, 255, 0), -1)
                                for i in [13, 14, 78, 308]:
                                    cv2.circle(frame, get_landmark_point(face_landmarks.landmark, i, iw, ih), 2, (255, 0, 0), -1)
                                
                                p10 = get_landmark_point(face_landmarks.landmark, 10, iw, ih)
                                p152 = get_landmark_point(face_landmarks.landmark, 152, iw, ih)
                                cv2.circle(frame, p10, 3, (255, 200, 0), -1)
                                cv2.circle(frame, p152, 3, (255, 0, 255), -1)
                                cv2.line(frame, p10, p152, (255, 255, 0), 1)
                                
                                ear, left_ear, right_ear = calculate_ear(face_landmarks.landmark, iw, ih)
                                mar = calculate_mar(face_landmarks.landmark, iw, ih)
                                pitch = calculate_head_pose_improved(face_landmarks.landmark, iw, ih)
                                
                                ear_history.append(ear)
                                mar_history.append(mar)
                                pitch_history.append(pitch)
                                
                                if state == "CALIBRATION":
                                    cv2.rectangle(frame, (w//2-250, h//2-130), (w//2+250, h//2+200), (30, 30, 30), -1)
                                    cv2.rectangle(frame, (w//2-250, h//2-130), (w//2+250, h//2+200), COLOR_WARN, 2)
                                    cv2.putText(frame, "KALIBRASI WAJAH", (w//2-80, h//2-100), FONT, FONT_NORMAL, COLOR_WARN, 2, cv2.LINE_AA)
                                    
                                    instructions = [
                                        "1. Hadap kamera dengan posisi normal",
                                        "2. Mata terbuka normal",
                                        "3. Mulut tertutup",
                                        "4. Kepala lurus ke depan",
                                        "5. Tekan 'C' untuk mulai (3 detik)",
                                        "6. Tunggu hingga selesai"
                                    ]
                                    for i, text in enumerate(instructions):
                                        cv2.putText(frame, text, (w//2-230, h//2 - 60 + i*22), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    
                                    rt_y = h//2 + 90
                                    cv2.putText(frame, "=== NILAI REAL-TIME ===", (w//2-230, rt_y), FONT, FONT_SMALL, COLOR_DEBUG, 1, cv2.LINE_AA)
                                    cv2.putText(frame, f"EAR: {ear:.3f}", (w//2-230, rt_y+20), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    cv2.putText(frame, f"MAR: {mar:.3f}", (w//2-230, rt_y+40), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (w//2-230, rt_y+60), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    cv2.putText(frame, f"FPS: {int(fps)}", (w//2-230, rt_y+80), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    
                                    key = cv2.waitKey(1) & 0xFF
                                    if key == ord('c'):
                                        state = "CALIBRATING"
                                        calibration_data = []
                                        calibration_start_time = time.time()
                                        print("[INFO] Memulai kalibrasi manual...")
                                
                                elif state == "CALIBRATING":
                                    elapsed_calibration = time.time() - calibration_start_time
                                    countdown = max(0, 3 - int(elapsed_calibration))
                                    
                                    cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), (30, 30, 30), -1)
                                    cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), COLOR_SAFE, 2)
                                    cv2.putText(frame, "KALIBRASI BERJALAN", (w//2-70, h//2-70), FONT, FONT_NORMAL, COLOR_SAFE, 2, cv2.LINE_AA)
                                    cv2.putText(frame, f"Sisa: {countdown} detik", (w//2-50, h//2-30), FONT, FONT_NORMAL, COLOR_INFO, 2, cv2.LINE_AA)
                                    
                                    progress_filled = int((elapsed_calibration / 3.0) * 200)
                                    cv2.rectangle(frame, (w//2-100, h//2+10), (w//2-100 + progress_filled, h//2+20), COLOR_SAFE, -1)
                                    
                                    calibration_data.append({'ear': ear, 'mar': mar, 'pitch': pitch})
                                    
                                    cv2.putText(frame, f"Sampel: {len(calibration_data)} frame", (w//2-90, h//2+40), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
                                    
                                    if elapsed_calibration >= 3.0 and len(calibration_data) > 0:
                                        baseline_ear = np.mean([d['ear'] for d in calibration_data])
                                        baseline_mar = np.mean([d['mar'] for d in calibration_data])
                                        baseline_pitch = np.median([d['pitch'] for d in calibration_data]) # Gunakan median lebih stabil
                                        
                                        THRESH_EAR_CLOSE = min(baseline_ear * 0.70, 0.22)
                                        THRESH_EAR_BLINK = THRESH_EAR_CLOSE * 1.15
                                        THRESH_MAR_YAWN = max(0.20, baseline_mar * 1.3)
                                        
                                        baseline_values = {'ear': baseline_ear, 'mar': baseline_mar, 'pitch': baseline_pitch}
                                        print(f"[INFO] Kalibrasi selesai! Base PITCH: {baseline_pitch:.1f}")
                                        
                                        detector = DrowsinessDetector(fps=fps, baseline_pitch=baseline_pitch)
                                        state = "MONITORING"
                                        calibration_data.clear()
                                
                                elif state == "MONITORING" and detector is not None:
                                    result = detector.update(ear, mar, pitch, fps)
                                    
                                    # === HUD OVERLAY DESIGN ===
                                    overlay = frame.copy()
                                    hud_w_left = 320
                                    hud_w_right = 360
                                    
                                    # Left Panel background
                                    cv2.rectangle(overlay, (0, 35), (hud_w_left, h-22), (0, 0, 0), -1)
                                    # Right Panel background
                                    cv2.rectangle(overlay, (w - hud_w_right, 35), (w, h-22), (0, 0, 0), -1)
                                    
                                    alpha = 0.5
                                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                                    
                                    pitch_color = COLOR_INFO
                                    if result['head_down_raw']: pitch_color = COLOR_DANGER
                                    elif result['head_up']: pitch_color = COLOR_WARN
                                    
                                    info_lines = [
                                        ("=== SENSOR DATA ===", COLOR_INFO),
                                        (f"EAR: {result['ear']:.3f}", COLOR_DANGER if result['eye_closed'] else COLOR_SAFE),
                                        (f"MAR: {result['mar']:.3f}", COLOR_DANGER if result['yawning'] else COLOR_SAFE),
                                        (f"PITCH: {result['pitch']:.1f} deg", pitch_color),
                                        (f"Deviasi: {result['pitch_deviation']:.1f} deg", pitch_color),
                                        ("", COLOR_INFO),
                                        ("=== THRESHOLD ===", COLOR_INFO),
                                        (f"EAR Th: {THRESH_EAR_CLOSE:.3f}", COLOR_DEBUG),
                                        (f"MAR Th: {THRESH_MAR_YAWN:.3f}", COLOR_DEBUG),
                                        (f"Pitch Base: {baseline_values['pitch']:.1f}", COLOR_DEBUG),
                                        ("", COLOR_INFO),
                                        ("=== DEBUG YAWNING ===", COLOR_INFO),
                                        (f"State: {result['yawn_state']}", COLOR_WARN if result['yawn_state'] != 'IDLE' else COLOR_INFO),
                                        (f"Hold: {result['yawn_hold_frames']}f | Pk: {result['yawn_peak_mar']:.2f}", COLOR_INFO),
                                        (f"Talking: {'YA' if result['is_talking'] else 'TIDAK'}", COLOR_INFO),
                                        ("", COLOR_INFO),
                                        ("=== PERCLOS ===", COLOR_INFO),
                                        (f"PERCLOS: {result['perclos']:.1f}%", COLOR_WARN if result['perclos']>20 else COLOR_SAFE),
                                        (f"Win: {result['window_frames']}f | Cls: {result['closed_frames']}f", COLOR_INFO),
                                        ("", COLOR_INFO),
                                        ("=== COUNTERS ===", COLOR_INFO),
                                        (f"Blinks: {result['blink_count']} (Val:{result['valid_blink_count']})", COLOR_INFO),
                                        (f"Yawns: {result['total_yawns']}", COLOR_INFO),
                                    ]
                                    
                                    panel_x, panel_y, line_height = 10, 55, 23
                                    for i, (text, col) in enumerate(info_lines):
                                        cv2.putText(frame, text, (panel_x, panel_y + i*line_height), FONT, FONT_SMALL, col, 1, cv2.LINE_AA)
                                    
                                    # === RIGHT PANEL DATA ===
                                    status_x = w - hud_w_right + 15
                                    status_y = 50
                                    
                                    if result['microsleep']: st_txt, st_col = "MICROSLEEP", COLOR_DANGER
                                    elif result['score'] >= 70: st_txt, st_col = "BAHAYA", COLOR_DANGER
                                    elif result['score'] >= 50: st_txt, st_col = "WASPADA", COLOR_WARN
                                    elif result['score'] >= 30: st_txt, st_col = "HATI-HATI", (0, 255, 255)
                                    else: st_txt, st_col = "AMAN", COLOR_SAFE
                                        
                                    # Status Box
                                    cv2.rectangle(frame, (status_x, status_y), (w-15, status_y+65), st_col, 1)
                                    cv2.putText(frame, "SYSTEM STATUS", (status_x+5, status_y+15), FONT, FONT_SMALL, st_col, 1, cv2.LINE_AA)
                                    cv2.putText(frame, st_txt, (status_x+5, status_y+40), FONT, 0.65, st_col, 2, cv2.LINE_AA)
                                    cv2.putText(frame, f"Score: {int(result['score'])}/100", (status_x+5, status_y+60), FONT, FONT_SMALL, st_col, 1, cv2.LINE_AA)
                                    
                                    # ROI Preview
                                    roi_y = status_y + 80
                                    if roi_frame is not None:
                                        roi_size = 180
                                        roi_resized = cv2.resize(roi_frame, (roi_size, roi_size))
                                        center_offset = (hud_w_right - roi_size) // 2
                                        roi_x = w - hud_w_right + center_offset
                                        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), COLOR_ROI, 1)
                                        frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size] = roi_resized
                                        cv2.putText(frame, "ROI PREVIEW", (roi_x, roi_y - 8), FONT, FONT_SMALL, COLOR_ROI, 1, cv2.LINE_AA)
                                    
                                    # Graphs
                                    graph_y_start = roi_y + 175
                                    COLOR_MAR_GRAPH = (0, 255, 255) # Yellow
                                    COLOR_PITCH_GRAPH = (0, 165, 255) # Orange
                                    
                                    for idx, (h_list, th, name, col) in enumerate([
                                        (ear_history, THRESH_EAR_CLOSE, "EAR History", COLOR_SAFE), 
                                        (mar_history, THRESH_MAR_YAWN, "MAR History", COLOR_MAR_GRAPH),
                                        (pitch_history, baseline_values['pitch'], "Pitch History", COLOR_PITCH_GRAPH)
                                    ]):
                                        g_y = graph_y_start + idx*75
                                        cv2.rectangle(frame, (status_x, g_y), (w-15, g_y+50), col, 1)
                                        cv2.putText(frame, name, (status_x+5, g_y-7), FONT, FONT_SMALL, col, 1, cv2.LINE_AA)
                                        
                                        if len(h_list) > 1:
                                            mx, mn = max(h_list) if h_list else 1, min(h_list) if h_list else 0
                                            rg = mx - mn if mx > mn else 1.0
                                            gw = hud_w_right - 30
                                            pts = [(status_x + int((i/len(h_list))*gw), int(g_y + 50 - ((v - mn)/rg)*50)) for i, v in enumerate(h_list)]
                                            pts = [(x, max(g_y, min(y, g_y+50))) for x, y in pts]
                                            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)
                                            # Draw threshold
                                            th_y = int(g_y + 50 - ((th - mn)/rg)*50)
                                            if g_y <= th_y <= g_y+50:
                                                cv2.line(frame, (status_x, th_y), (w-15, th_y), (100, 100, 100), 1, cv2.LINE_AA)

                                    # Center Screen Warnings (Outlined)
                                    if result['microsleep']:
                                        if int(time.time() * 4) % 2 == 0:
                                            over = frame.copy(); cv2.rectangle(over, (0,0), (w,h), (0,0,255), -1)
                                            cv2.addWeighted(over, 0.15, frame, 0.85, 0, frame)
                                        draw_text_with_outline(frame, "!!! MICROSLEEP DETECTED !!!", (w//2-180, h//2), FONT, 0.7, COLOR_DANGER, 2)
                                    elif result['score'] >= 70:
                                        if int(time.time() * 2) % 2 == 0:
                                            over = frame.copy(); cv2.rectangle(over, (0,0), (w,h), (0,0,255), -1)
                                            cv2.addWeighted(over, 0.1, frame, 0.9, 0, frame)
                                        draw_text_with_outline(frame, "!!! PERINGATAN KELELAHAN !!!", (w//2-180, h//2), FONT, 0.7, COLOR_DANGER, 2)
                                    elif result['score'] >= 50:
                                        draw_text_with_outline(frame, "WASPADA - Tanda Kelelahan", (w//2-150, h//2-50), FONT, 0.6, COLOR_WARN, 2)
                                        
                                    if result['yawning']:
                                        draw_text_with_outline(frame, "MENGUAP TERDETEKSI", (w//2-120, h//2+30), FONT, 0.6, (0, 200, 255), 2)
                                    if result['head_down_raw']:
                                        draw_text_with_outline(frame, "KEPALA MENUNDUK", (w//2-100, h//2+60), FONT, 0.6, COLOR_DANGER, 2)
                                        
            if not results_detection.detections:
                cv2.rectangle(frame, (w//2-230, h//2-30), (w//2+230, h//2+30), (30, 30, 30), -1)
                cv2.rectangle(frame, (w//2-230, h//2-30), (w//2+230, h//2+30), COLOR_WARN, 1)
                draw_text_with_outline(frame, "MENCARI WAJAH...", (w//2-130, h//2+10), FONT, 0.7, COLOR_WARN, 2)

            cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(frame, f"Raspberry Pi 5 DROWSINESS | FPS: {int(fps)} | State: {state}", (10, 22), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, h-22), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, "q:Keluar | c:Kalibrasi | r:Reset | s:Screenshot | f:Flip", (10, h-5), FONT, FONT_SMALL, COLOR_INFO, 1, cv2.LINE_AA)
            
            cv2.imshow('Deteksi Kelelahan RPi5 (TANPA AUDIO - rpicam-vid)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                flip_mode = -1 if flip_mode == 0 else 0
            elif key == ord('r'):
                state = "CALIBRATION"
                detector = None
                calibration_data.clear()
                ear_history.clear()
                mar_history.clear()
                pitch_history.clear()
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_rpi_{time.strftime('%Y%m%d_%H%M%S')}.png", frame)
    except BaseException as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Process dihentikan: {e}")
    finally:
        print("[INFO] Membersihkan resource...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
