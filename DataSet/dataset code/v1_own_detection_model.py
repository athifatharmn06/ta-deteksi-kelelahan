import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from collections import deque
import sys
import math
from pathlib import Path

# ============================================
# KONFIGURASI SISTEM
# ============================================
# Bobot deteksi
W_EYE = 0.45        # Bobot mata
W_MOUTH = 0.25      # Bobot mulut
W_HEAD = 0.30       # Bobot kepala

# Threshold default
THRESH_EAR_CLOSE = 0.18      # EAR untuk mata tertutup
THRESH_EAR_BLINK = 0.22      # EAR untuk kedipan
THRESH_MAR_YAWN = 0.30       # MAR untuk menguap
THRESH_HEAD_PITCH = 4.0      # Pitch untuk nunduk (derajat)

# Parameter waktu
TIME_MICROSLEEP_LIMIT = 1.5  # Microsleep (3.0 detik)
TIME_YAWN_MIN_DURATION = 0.5  # Menguap minimal 0.5 detik (turun dari 0.8 untuk tangkap yawn pendek)
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
# INISIALISASI MEDIAPIPE (BLAZEFACE + FACE MESH)
# ============================================
print("[INFO] Memulai Inisialisasi MediaPipe...")
print("[INFO] Menggunakan BlazeFace untuk ROI detection...")

# BlazeFace untuk deteksi wajah cepat
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# Face Mesh untuk landmark (hanya di ROI)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================
# INISIALISASI AUDIO
# ============================================
print("[INFO] Memulai Inisialisasi Audio...")
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
audio_available = False

# Load semua file audio
sound_warning = None
sound_microsleep = None

try:
    sound_warning = pygame.mixer.Sound('kelelahan ringan.wav')
    sound_microsleep = pygame.mixer.Sound('microsleep alert.wav')
    print("[INFO] File audio berhasil dimuat.")
    audio_available = True
except Exception as e:
    print(f"[WARNING] Gagal memuat audio: {e}")
    print("[INFO] Sistem akan berjalan tanpa suara.")

# ============================================
# FUNGSI UTILITAS GEOMETRI
# ============================================
def euclidean_distance(p1, p2):
    """Menghitung jarak Euclidean antara dua titik"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_landmark_point(landmark, index, width, height):
    """Mendapatkan koordinat landmark dalam piksel"""
    return (int(landmark[index].x * width), int(landmark[index].y * height))

def calculate_ear_single_eye(eye_points):
    """Menghitung EAR untuk satu mata dari 6 titik"""
    # Jarak vertikal
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    
    # Jarak horizontal
    C = euclidean_distance(eye_points[0], eye_points[3])
    
    if C == 0:
        return 0.0
    
    return (A + B) / (2.0 * C)

# ============================================
# FUNGSI EAR (EYE ASPECT RATIO) - OPTIMIZED
# ============================================
def calculate_ear(landmarks, width, height):
    """Menghitung Eye Aspect Ratio dengan 6 titik per mata"""
    # Titik untuk mata kiri
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    left_eye_points = [get_landmark_point(landmarks, i, width, height) for i in left_eye_indices]
    
    # Titik untuk mata kanan
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_points = [get_landmark_point(landmarks, i, width, height) for i in right_eye_indices]
    
    left_ear = calculate_ear_single_eye(left_eye_points)
    right_ear = calculate_ear_single_eye(right_eye_points)
    
    avg_ear = (left_ear + right_ear) / 2.0
    return avg_ear, left_ear, right_ear

# ============================================
# FUNGSI MAR (MOUTH ASPECT RATIO)
# ============================================
def calculate_mar(landmarks, width, height):
    """Menghitung Mouth Aspect Ratio dengan lebih banyak titik untuk akurasi"""
    # Titik bibir atas
    upper_points = [
        get_landmark_point(landmarks, 13, width, height),
        get_landmark_point(landmarks, 0, width, height),
        get_landmark_point(landmarks, 17, width, height)
    ]
    
    # Titik bibir bawah
    lower_points = [
        get_landmark_point(landmarks, 14, width, height),
        get_landmark_point(landmarks, 18, width, height),
        get_landmark_point(landmarks, 267, width, height)
    ]
    
    # Titik sudut kiri dan kanan
    left_corner = get_landmark_point(landmarks, 78, width, height)
    right_corner = get_landmark_point(landmarks, 308, width, height)
    
    # Hitung titik tengah atas dan bawah
    upper_center = (
        sum(p[0] for p in upper_points) // len(upper_points),
        sum(p[1] for p in upper_points) // len(upper_points)
    )
    
    lower_center = (
        sum(p[0] for p in lower_points) // len(lower_points),
        sum(p[1] for p in lower_points) // len(lower_points)
    )
    
    # Jarak vertikal dan horizontal
    vertical = euclidean_distance(upper_center, lower_center)
    horizontal = euclidean_distance(left_corner, right_corner)
    
    if horizontal == 0:
        return 0.0
    
    return vertical / horizontal

# ============================================
# FUNGSI HEAD POSE
# ============================================
def calculate_head_pose_improved(landmarks, width, height):
    """Menghitung head pose dengan metode yang lebih akurat"""
    try:
        # Titik-titik kunci
        forehead = get_landmark_point(landmarks, 10, width, height)
        chin = get_landmark_point(landmarks, 152, width, height)
        nose = get_landmark_point(landmarks, 1, width, height)
        left_eye = get_landmark_point(landmarks, 33, width, height)
        right_eye = get_landmark_point(landmarks, 263, width, height)
        
        # Hitung panjang dan lebar wajah
        face_height = abs(chin[1] - forehead[1])
        face_width = abs(right_eye[0] - left_eye[0])
        
        if face_height == 0 or face_width == 0:
            return 0.0
        
        # Hitung posisi relatif hidung
        nose_to_forehead = abs(nose[1] - forehead[1])
        nose_rel_position = nose_to_forehead / face_height
        
        # Konversi ke pitch
        pitch_from_nose = (0.5 - nose_rel_position) * 100
        
        # Hitung dari perbandingan dahi-dagu
        dx = chin[0] - forehead[0]
        dy = chin[1] - forehead[1]
        
        if dy != 0:
            angle_from_vertical = math.degrees(math.atan2(dx, abs(dy)))
        else:
            angle_from_vertical = 0
        
        if dy > 0:
            pitch_from_chin = -angle_from_vertical
        else:
            pitch_from_chin = angle_from_vertical
        
        # Gabungkan kedua perhitungan
        pitch = pitch_from_nose * 0.4 + pitch_from_chin * 0.6
        
        # Batasi range
        pitch = max(-45, min(45, pitch))
        
        return pitch
        
    except Exception as e:
        print(f"[WARNING] Error in head pose: {e}")
        return 0.0

# ============================================
# PERCLOS CALCULATOR
# ============================================
class PERCLOSCalculator:
    """Kalkulator PERCLOS"""
    def __init__(self, window_seconds=60, fps=30):
        self.window_seconds = window_seconds
        self.fps = fps
        self.window_size = int(window_seconds * fps)
        
        # Buffer untuk menyimpan status mata
        self.eye_states = deque(maxlen=self.window_size)
        
        # Variabel statistik
        self.total_frames = 0
        self.closed_frames = 0
        self.last_perclos = 0.0
        
        # Untuk deteksi kedipan
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_rate = 0
        self.last_eye_state = False
        
    def update(self, ear, threshold):
        """Update status mata dan hitung PERCLOS"""
        # Deteksi status mata
        is_closed = ear < threshold
        
        # Deteksi transisi untuk blink counting
        if self.last_eye_state == True and is_closed == False:
            self.blink_count += 1
        self.last_eye_state = is_closed
        
        # Simpan status
        self.eye_states.append(is_closed)
        self.total_frames = len(self.eye_states)
        
        # Hitung frame tertutup
        self.closed_frames = sum(1 for state in self.eye_states if state)
        
        # Hitung PERCLOS
        if self.total_frames > 0:
            self.last_perclos = (self.closed_frames / self.total_frames) * 100.0
        
        # Hitung blink rate
        current_time = time.time()
        time_interval = current_time - self.last_blink_time
        
        # Reset blink rate setiap menit
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
    """Detektor kantuk dengan filter multi-modal"""
    def __init__(self, fps=30, baseline_pitch=0.0):
        # Filter untuk masing-masing parameter
        self.ear_buffer = deque(maxlen=5)
        self.mar_buffer = deque(maxlen=5)
        self.pitch_buffer = deque(maxlen=5)
        
        # State variables
        self.eye_closed_frames = 0
        self.yawn_frames = 0
        self.head_down_frames = 0
        
        # Timing variables
        self.last_blink_time = time.time()
        self.microsleep_start = None
        self.yawn_start = None
        self.head_down_start = None
        
        # Counters
        self.blink_count = 0
        self.yawn_count = 0
        self.total_blinks = 0
        self.total_yawns = 0
        
        # Variabel untuk yawning detection
        self.normal_ear = None
        self.normal_mar = None
        self.last_mar = 0.0
        self.mar_increasing = False
        
        # Audio cooldowns
        self.last_warning_audio = 0
        self.last_microsleep_audio = 0
        
        # PERCLOS Calculator
        self.perclos_calc = PERCLOSCalculator(window_seconds=TIME_PERCLOS_WINDOW, fps=fps)
        
        # State flags
        self.is_yawning = False
        self.was_yawning = False
        
        # Cooldown setelah yawn selesai — cegah 1 yawn fisik terhitung 2x
        self.yawn_cooldown = 0
        self.YAWN_COOLDOWN_FRAMES = int(3.0 * fps)  # 3 detik cooldown
        
        # Baseline pitch untuk referensi
        self.baseline_pitch = baseline_pitch
        
        # Pitch threshold
        self.pitch_threshold = THRESH_HEAD_PITCH
        
    def smooth_value(self, buffer, new_value):
        """Smoothing dengan moving average"""
        buffer.append(new_value)
        return sum(buffer) / len(buffer)
    
    def update(self, ear, mar, pitch, fps=30):
        """Update detektor dengan nilai terbaru"""
        # Smooth values
        ear_smooth = self.smooth_value(self.ear_buffer, ear)
        mar_smooth = self.smooth_value(self.mar_buffer, mar)
        pitch_smooth = self.smooth_value(self.pitch_buffer, pitch)
        
        # Update normal values jika belum diatur
        if self.normal_ear is None:
            self.normal_ear = ear_smooth
        if self.normal_mar is None:
            self.normal_mar = mar_smooth
        
        # Deteksi apakah MAR sedang meningkat
        if mar_smooth > self.last_mar * 1.05:
            self.mar_increasing = True
        elif mar_smooth < self.last_mar * 0.95:
            self.mar_increasing = False
        self.last_mar = mar_smooth
        
        # Update PERCLOS
        perclos_data = self.perclos_calc.update(ear_smooth, THRESH_EAR_CLOSE)
        
        # Update total blinks
        self.blink_count = perclos_data['blink_count']
        
        # Deteksi kondisi DASAR
        eye_closed = ear_smooth < THRESH_EAR_CLOSE
        
        # HEAD POSE
        pitch_deviation = pitch_smooth - self.baseline_pitch
        
        # Menunduk
        head_down = pitch_deviation < -self.pitch_threshold and pitch_smooth < -3.0
        
        # Menengadah
        head_up = pitch_deviation > self.pitch_threshold and pitch_smooth > 3.0
        
        # DETEKSI YAWNING
        mar_threshold = THRESH_MAR_YAWN
        
        # Kondisi utama untuk yawning
        mar_condition = mar_smooth > mar_threshold
        
        # Kondisi tambahan: MAR harus meningkat dari normal
        if self.normal_mar is not None:
            mar_increased = mar_smooth > self.normal_mar * 1.2
        else:
            mar_increased = True
        
        # Bukan sedang mata tertutup penuh
        not_fully_closed = not eye_closed
        
        # Yawning candidate
        yawning_candidate = mar_condition and mar_increased and not_fully_closed
        
        # Update yawning state
        if yawning_candidate and self.yawn_cooldown == 0:
            self.yawn_frames = min(self.yawn_frames + 6, int(TIME_YAWN_MIN_DURATION * fps * 2))
        else:
            # Decay lambat (-1) agar gap kecil di tengah yawn tidak putus event
            self.yawn_frames = max(self.yawn_frames - 1, 0)
        
        # Turunkan cooldown setiap frame
        if self.yawn_cooldown > 0:
            self.yawn_cooldown -= 1
        
        # Yawn terdeteksi jika sudah mencapai durasi minimal
        # Blokir deteksi baru jika masih dalam cooldown
        yawn_detected = (self.yawn_frames >= int(TIME_YAWN_MIN_DURATION * fps)
                         and self.yawn_cooldown == 0)
        
        # Update state yawning
        if yawn_detected and not self.was_yawning:
            self.is_yawning = True
            self.yawn_count += 1
            self.total_yawns += 1
        elif not yawn_detected and self.was_yawning:
            # Yawn baru saja selesai → mulai cooldown 3 detik
            self.is_yawning = False
            self.yawn_cooldown = self.YAWN_COOLDOWN_FRAMES
            self.yawn_frames = 0  # reset paksa supaya tidak langsung aktif lagi
        elif not yawn_detected:
            self.is_yawning = False
        
        self.was_yawning = yawn_detected
        
        # MICROSLEEP DETECTION
        if eye_closed and not self.is_yawning and mar_smooth < mar_threshold * 0.7:
            self.eye_closed_frames = min(self.eye_closed_frames + 1, TIME_CONSECUTIVE_FRAMES * 2)
        else:
            self.eye_closed_frames = max(self.eye_closed_frames - 2, 0)
        
        # Microsleep hanya jika mata tertutup cukup lama dan BUKAN yawning
        microsleep_detected = False
        if (self.eye_closed_frames >= TIME_CONSECUTIVE_FRAMES and 
            not self.is_yawning and 
            mar_smooth < mar_threshold * 0.6):
            
            if self.microsleep_start is None:
                self.microsleep_start = time.time()
            elif time.time() - self.microsleep_start > TIME_MICROSLEEP_LIMIT:
                microsleep_detected = True
        else:
            self.microsleep_start = None
        
        # Head down detection
        if head_down:
            self.head_down_frames = min(self.head_down_frames + 1, TIME_CONSECUTIVE_FRAMES)
        else:
            self.head_down_frames = max(self.head_down_frames - 1, 0)
        
        head_down_detected = self.head_down_frames >= TIME_CONSECUTIVE_FRAMES
        
        # Update normal values secara adaptif
        if (not yawn_detected and not eye_closed and 
            mar_smooth < mar_threshold * 0.5 and
            not head_down and not head_up):
            if self.normal_ear is not None:
                self.normal_ear = self.normal_ear * 0.98 + ear_smooth * 0.02
            if self.normal_mar is not None:
                self.normal_mar = self.normal_mar * 0.98 + mar_smooth * 0.02
        
        # Hitung skor multi-modal
        score = 0
        
        # Komponen mata (45%)
        if microsleep_detected:
            score += W_EYE * 100
        elif self.eye_closed_frames >= TIME_CONSECUTIVE_FRAMES and not self.is_yawning:
            score += W_EYE * 60
        elif perclos_data['perclos'] > 20:
            score += W_EYE * (min(perclos_data['perclos'], 40) / 40 * 70)
        
        # Komponen mulut (25%)
        if yawn_detected:
            score += W_MOUTH * 70
        elif self.yawn_frames > int(TIME_YAWN_MIN_DURATION * fps * 0.3):
            score += W_MOUTH * 20
        
        # Komponen kepala (30%)
        if head_down_detected:
            score += W_HEAD * 100
        elif self.head_down_frames > TIME_CONSECUTIVE_FRAMES * 0.5:
            score += W_HEAD * 40
        elif head_up:
            score += W_HEAD * 30
        
        # Normalisasi skor
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
            'total_blinks': self.total_blinks,
            'total_yawns': self.total_yawns,
            'window_frames': perclos_data['window_frames'],
            'closed_frames': perclos_data['closed_frames'],
            'normal_ear': self.normal_ear,
            'normal_mar': self.normal_mar,
            'is_yawning_state': self.is_yawning,
            'baseline_pitch': self.baseline_pitch,
            'pitch_threshold': self.pitch_threshold
        }

# ============================================
# AUDIO MANAGER
# ============================================
class AudioManager:
    """Mengatur pemutaran audio tanpa tumpang tindih"""
    def __init__(self):
        self.audio_cooldowns = {
            'warning': 5.0,
            'microsleep': 8.0
        }
        
        self.last_play_time = {
            'warning': 0,
            'microsleep': 0
        }
        
        self.is_playing = False
        self.channels = {}
    
    def can_play(self, audio_type):
        """Cek apakah audio bisa diputar"""
        current_time = time.time()
        
        # Cek cooldown
        if current_time - self.last_play_time[audio_type] < self.audio_cooldowns[audio_type]:
            return False
        
        # Cek apakah ada audio yang sedang diputar
        if self.is_playing:
            return False
        
        return True
    
    def play_audio(self, audio_type, sound):
        """Mainkan audio dengan type tertentu"""
        if not audio_available:
            return False
        
        if self.can_play(audio_type):
            try:
                # Hentikan channel sebelumnya jika ada
                if audio_type in self.channels:
                    try:
                        self.channels[audio_type].stop()
                    except:
                        pass
                
                # Mainkan suara
                channel = sound.play()
                
                # Simpan channel
                self.channels[audio_type] = channel
                self.last_play_time[audio_type] = time.time()
                self.is_playing = True
                
                # Reset is_playing setelah durasi audio
                audio_duration = 2.0
                try:
                    if hasattr(sound, 'get_length'):
                        audio_duration = sound.get_length()
                except:
                    pass
                
                # Buat thread untuk reset
                import threading
                def reset_playing():
                    time.sleep(audio_duration + 0.1)
                    self.is_playing = False
                
                thread = threading.Thread(target=reset_playing)
                thread.daemon = True
                thread.start()
                
                return True
            except Exception as e:
                print(f"[WARNING] Gagal memutar audio {audio_type}: {e}")
                self.is_playing = False
                return False
        return False

# ============================================
# EVENT RECORDER + CSV EXPORT
# ============================================
class EventRecorder:
    """Merekam history event (yawning, microsleep) dan PERCLOS per detik untuk diekspor ke CSV"""

    def __init__(self, video_path, fps):
        self.video_path  = video_path
        self.fps         = fps
        self.source_name = Path(video_path).stem

        # Event tracking
        self._yawn_active       = False
        self._yawn_start_frame  = None
        self._micro_active      = False
        self._micro_start_frame = None

        self.event_rows   = []   # untuk events.csv
        self.perclos_rows = []   # untuk perclos.csv
        self._last_perclos_second = -1

    @staticmethod
    def _ts(frame, fps):
        total_ms = int((frame / fps) * 1000)
        m  = total_ms // 60000
        s  = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        return f"{m:02d}:{s:02d}.{ms:03d}"

    def update(self, frame_idx, result):
        """Dipanggil setiap frame setelah detector.update()"""
        fps = self.fps

        # ── Yawning event ──────────────────────────────────────────────────
        if result['yawning'] and not self._yawn_active:
            self._yawn_active      = True
            self._yawn_start_frame = frame_idx
        elif not result['yawning'] and self._yawn_active:
            self._yawn_active = False
            fs = self._yawn_start_frame
            fe = frame_idx - 1
            self.event_rows.append({
                'source':       self.source_name,
                'event_type':   'yawning',
                'time_start':   self._ts(fs, fps),
                'time_end':     self._ts(fe, fps),
                'duration_sec': round((fe - fs + 1) / fps, 3),
                'frame_start':  fs,
                'frame_end':    fe,
            })

        # ── Microsleep event ───────────────────────────────────────────────
        if result['microsleep'] and not self._micro_active:
            self._micro_active      = True
            self._micro_start_frame = frame_idx
        elif not result['microsleep'] and self._micro_active:
            self._micro_active = False
            fs = self._micro_start_frame
            fe = frame_idx - 1
            self.event_rows.append({
                'source':       self.source_name,
                'event_type':   'microsleep',
                'time_start':   self._ts(fs, fps),
                'time_end':     self._ts(fe, fps),
                'duration_sec': round((fe - fs + 1) / fps, 3),
                'frame_start':  fs,
                'frame_end':    fe,
            })

        # ── PERCLOS per detik ──────────────────────────────────────────────
        current_second = int(frame_idx / fps)
        if current_second != self._last_perclos_second:
            self._last_perclos_second = current_second
            m  = current_second // 60
            s  = current_second % 60
            self.perclos_rows.append({
                'source':        self.source_name,
                'time':          f"{m:02d}:{s:02d}.000",
                'second':        current_second,
                'perclos':       round(result['perclos'], 4),
                'frames_closed': result.get('closed_frames', ''),
                'window_frames': result.get('window_frames', ''),
                'ear':           round(result['ear'], 4),
                'score':         round(result['score'], 2),
            })

    def _flush_open_events(self, last_frame):
        """Tutup event yang masih aktif saat video selesai"""
        if self._yawn_active:
            fs = self._yawn_start_frame
            fe = last_frame
            self.event_rows.append({
                'source': self.source_name, 'event_type': 'yawning',
                'time_start': self._ts(fs, self.fps), 'time_end': self._ts(fe, self.fps),
                'duration_sec': round((fe-fs+1)/self.fps, 3),
                'frame_start': fs, 'frame_end': fe,
            })
        if self._micro_active:
            fs = self._micro_start_frame
            fe = last_frame
            self.event_rows.append({
                'source': self.source_name, 'event_type': 'microsleep',
                'time_start': self._ts(fs, self.fps), 'time_end': self._ts(fe, self.fps),
                'duration_sec': round((fe-fs+1)/self.fps, 3),
                'frame_start': fs, 'frame_end': fe,
            })

    def save_csv(self, output_dir, last_frame):
        import csv
        self._flush_open_events(last_frame)
        out = Path(output_dir)

        # Sort events by time
        self.event_rows.sort(key=lambda x: x['frame_start'])

        # events.csv
        events_path = out / "own_detection_model_events.csv"
        ev_fields   = ['source','event_type','time_start','time_end','duration_sec','frame_start','frame_end']
        with open(events_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=ev_fields)
            w.writeheader(); w.writerows(self.event_rows)
        print(f"[CSV] events.csv  → {len(self.event_rows)} events  ({events_path})")

        # perclos.csv
        perclos_path = out / "own_detection_model_perclos.csv"
        pc_fields    = ['source','time','second','perclos','frames_closed','window_frames','ear','score']
        with open(perclos_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=pc_fields)
            w.writeheader(); w.writerows(self.perclos_rows)
        print(f"[CSV] perclos.csv → {len(self.perclos_rows)} seconds ({perclos_path})")


# ============================================
# FUNGSI UTAMA UNTUK VIDEO FILE
# ============================================
def main():
    print("[INFO] Sistem Deteksi Kelelahan untuk Video File")
    print("[INFO] Strategi: BlazeFace (ROI) → Face Mesh (Landmark di ROI)")
    
    # Path ke file video - GANTI INI DENGAN PATH VIDEO ANDA
    video_path = "C:\TA Deteksi Kelelahan\DataSet\DMD Dataset 1\gA_1_s5_2019-03-14T14;26;17+01;00_ir_face.mp4"  # <-- GANTI DENGAN PATH VIDEO ANDA
    
    print(f"[INFO] Membuka video: {video_path}")
    
    # Buka video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[FATAL] Tidak dapat membuka video: {video_path}")
        print("[INFO] Pastikan file video ada di direktori yang benar")
        sys.exit()
    
    # Dapatkan properti video
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_video if fps_video > 0 else 0
    
    print(f"[INFO] Resolusi: {w}x{h}")
    print(f"[INFO] FPS Video: {fps_video:.2f}")
    print(f"[INFO] Total Frame: {total_frames}")
    print(f"[INFO] Durasi: {duration:.2f} detik")

    # Inisialisasi EventRecorder
    event_recorder = EventRecorder(video_path, fps_video)
    
    # Inisialisasi audio manager
    audio_manager = AudioManager()
    
    # Variabel untuk kalibrasi otomatis
    calibration_data = []
    baseline_values = {}
    detector = None
    calibrated = False
    calibration_duration = 2.0  # Kalibrasi selama 2 detik pertama
    
    # State untuk tampilan
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    # History untuk grafik
    ear_history = deque(maxlen=100)
    mar_history = deque(maxlen=100)
    pitch_history = deque(maxlen=100)
    
    # Font settings
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = 0.35
    FONT_NORMAL = 0.45
    FONT_THICKNESS = 1
    
    print("\n[INFO] Sistem siap.")
    print(f"[INFO] Kalibrasi otomatis: {calibration_duration} detik pertama video")
    print("[INFO] Tekan 'q' untuk keluar, 's' untuk screenshot")
    
    # Hitung frame untuk kalibrasi
    calibration_frames_needed = int(calibration_duration * fps_video)
    current_frame = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame += 1
            frame_original = frame.copy()
            
            # Hitung FPS tampilan
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Hitung waktu video saat ini
            video_time = current_frame / fps_video
            
            # STEP 1: BLAZEFACE ROI DETECTION
            roi_frame = None
            roi_coords = None
            face_detected = False
            
            # Konversi ke RGB untuk BlazeFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Deteksi wajah dengan BlazeFace
            results_detection = face_detection.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            if results_detection.detections:
                for detection in results_detection.detections:
                    # Dapatkan bounding box dari BlazeFace
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    
                    # Konversi ke koordinat piksel
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w_bbox = int(bboxC.width * iw)
                    h_bbox = int(bboxC.height * ih)
                    
                    # Tambahkan padding untuk ROI
                    padding = 30
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(iw, x + w_bbox + padding)
                    y2 = min(ih, y + h_bbox + padding)
                    
                    roi_coords = (x1, y1, x2, y2)
                    
                    # Gambar bounding box BlazeFace
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)
                    cv2.putText(frame, "BlazeFace ROI", (x1, y1-10), 
                               FONT, FONT_SMALL, COLOR_ROI, FONT_THICKNESS)
                    
                    # Potong ROI untuk Face Mesh
                    roi_frame = frame_original[y1:y2, x1:x2]
                    face_detected = True
                    
                    # STEP 2: FACE MESH PADA ROI SAJA
                    if roi_frame is not None and roi_frame.shape[0] > 0 and roi_frame.shape[1] > 0:
                        roi_h, roi_w, _ = roi_frame.shape
                        
                        # Konversi ROI ke RGB untuk Face Mesh
                        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                        roi_rgb.flags.writeable = False
                        
                        # Jalankan Face Mesh hanya pada ROI
                        results_mesh = face_mesh.process(roi_rgb)
                        roi_rgb.flags.writeable = True
                        
                        if results_mesh.multi_face_landmarks:
                            for face_landmarks in results_mesh.multi_face_landmarks:
                                # Skala landmark kembali ke ukuran frame asli
                                for landmark in face_landmarks.landmark:
                                    landmark.x = (landmark.x * roi_w + x1) / iw
                                    landmark.y = (landmark.y * roi_h + y1) / ih
                                
                                # Gambar landmark mata dan mulut
                                # Mata kiri - hijau
                                left_eye_indices = [33, 160, 158, 133, 153, 144]
                                for i in left_eye_indices:
                                    pt = get_landmark_point(face_landmarks.landmark, i, iw, ih)
                                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                                
                                # Mata kanan - hijau
                                right_eye_indices = [362, 385, 387, 263, 373, 380]
                                for i in right_eye_indices:
                                    pt = get_landmark_point(face_landmarks.landmark, i, iw, ih)
                                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                                
                                # Mulut - biru
                                mouth_indices = [13, 14, 78, 308]
                                for i in mouth_indices:
                                    pt = get_landmark_point(face_landmarks.landmark, i, iw, ih)
                                    cv2.circle(frame, pt, 2, (255, 0, 0), -1)
                                
                                # Titik-titik kunci untuk head pose
                                # Dahi - biru muda
                                p10 = get_landmark_point(face_landmarks.landmark, 10, iw, ih)
                                cv2.circle(frame, p10, 3, (255, 200, 0), -1)
                                
                                # Dagu - merah muda
                                p152 = get_landmark_point(face_landmarks.landmark, 152, iw, ih)
                                cv2.circle(frame, p152, 3, (255, 0, 255), -1)
                                
                                # Hidung - kuning
                                p1 = get_landmark_point(face_landmarks.landmark, 1, iw, ih)
                                cv2.circle(frame, p1, 3, (0, 255, 255), -1)
                                
                                # Gambar garis untuk menunjukkan orientasi kepala
                                cv2.line(frame, p10, p152, (255, 255, 0), 1)
                                
                                # Hitung parameter
                                ear, left_ear, right_ear = calculate_ear(face_landmarks.landmark, iw, ih)
                                mar = calculate_mar(face_landmarks.landmark, iw, ih)
                                pitch = calculate_head_pose_improved(face_landmarks.landmark, iw, ih)
                                
                                # KALIBRASI OTOMATIS (2 detik pertama)
                                if video_time < calibration_duration and not calibrated:
                                    # Kumpulkan data kalibrasi
                                    calibration_data.append({
                                        'ear': ear,
                                        'mar': mar,
                                        'pitch': pitch
                                    })
                                    
                                    # Tampilkan status kalibrasi
                                    cv2.putText(frame, f"KALIBRASI: {video_time:.1f}/{calibration_duration} detik", 
                                               (w//2-100, 50), FONT, FONT_NORMAL, COLOR_WARN, 2)
                                    cv2.putText(frame, f"Sample: {len(calibration_data)}", 
                                               (w//2-50, 80), FONT, FONT_SMALL, COLOR_INFO, 1)
                                    
                                # Jika sudah melewati waktu kalibrasi dan belum dikalibrasi
                                elif video_time >= calibration_duration and not calibrated and len(calibration_data) > 0:
                                    # Hitung baseline dari data kalibrasi
                                    ear_values = [d['ear'] for d in calibration_data]
                                    mar_values = [d['mar'] for d in calibration_data]
                                    pitch_values = [d['pitch'] for d in calibration_data]
                                    
                                    baseline_ear = np.mean(ear_values)
                                    baseline_mar = np.mean(mar_values)
                                    baseline_pitch = np.mean(pitch_values)
                                    
                                    # Set threshold berdasarkan baseline
                                    global THRESH_EAR_CLOSE, THRESH_MAR_YAWN, THRESH_HEAD_PITCH
                                    
                                    THRESH_EAR_CLOSE = baseline_ear * 0.70
                                    THRESH_EAR_CLOSE = max(0.15, min(THRESH_EAR_CLOSE, 0.22))
                                    THRESH_EAR_BLINK = THRESH_EAR_CLOSE * 1.15
                                    
                                    # MAR threshold: 30% lebih tinggi dari baseline
                                    THRESH_MAR_YAWN = max(0.25, min(baseline_mar * 1.3, 0.35))
                                    
                                    # Pitch threshold tetap
                                    THRESH_HEAD_PITCH = 4.0
                                    
                                    # Simpan baseline values
                                    baseline_values = {
                                        'ear': baseline_ear,
                                        'mar': baseline_mar,
                                        'pitch': baseline_pitch
                                    }
                                    
                                    # Inisialisasi detektor
                                    detector = DrowsinessDetector(fps=fps_video, baseline_pitch=baseline_pitch)
                                    calibrated = True
                                    
                                    print(f"\n[INFO] Kalibrasi otomatis selesai!")
                                    print(f"  Baseline EAR: {baseline_ear:.3f} -> Th: {THRESH_EAR_CLOSE:.3f}")
                                    print(f"  Baseline MAR: {baseline_mar:.3f} -> Th: {THRESH_MAR_YAWN:.3f}")
                                    print(f"  Baseline Pitch: {baseline_pitch:.1f} deg")
                                    print(f"  Sampel: {len(calibration_data)} frame")
                                
                                # MONITORING SETELAH KALIBRASI
                                elif calibrated and detector is not None:
                                    # Simpan history untuk grafik
                                    ear_history.append(ear)
                                    mar_history.append(mar)
                                    pitch_history.append(pitch)
                                    
                                    # Update detektor
                                    result = detector.update(ear, mar, pitch, fps_video)
                                    
                                    # Rekam event dan PERCLOS untuk CSV
                                    event_recorder.update(current_frame, result)
                                    
                                    # PANEL INFORMASI KIRI
                                    panel_x = 10
                                    panel_y = 60
                                    line_height = 16
                                    
                                    # Tentukan warna untuk pitch
                                    pitch_color = COLOR_INFO
                                    if result['head_down_raw']:
                                        pitch_color = COLOR_DANGER
                                    elif result['head_up']:
                                        pitch_color = COLOR_WARN
                                    elif abs(result['pitch_deviation']) > THRESH_HEAD_PITCH * 0.7:
                                        pitch_color = COLOR_WARN
                                    
                                    # Hitung threshold aktual
                                    pitch_th_down = baseline_values['pitch'] - THRESH_HEAD_PITCH
                                    pitch_th_up = baseline_values['pitch'] + THRESH_HEAD_PITCH
                                    
                                    info_lines = [
                                        "=== SENSOR DATA ===",
                                        f"EAR: {result['ear']:.3f}",
                                        f"MAR: {result['mar']:.3f}",
                                        f"PITCH: {result['pitch']:.1f} deg",
                                        f"Deviasi: {result['pitch_deviation']:.1f} deg",
                                        "",
                                        "=== THRESHOLD ===",
                                        f"EAR Th: {THRESH_EAR_CLOSE:.3f}",
                                        f"MAR Th: {THRESH_MAR_YAWN:.3f}",
                                        f"Pitch Baseline: {baseline_values['pitch']:.1f} deg",
                                        f"Pitch Th Down: {pitch_th_down:.1f} deg",
                                        f"Pitch Th Up: {pitch_th_up:.1f} deg",
                                        "",
                                        "=== PERCLOS ===",
                                        f"PERCLOS: {result['perclos']:.1f}%",
                                        f"Window: {result['window_frames']}f",
                                        f"Closed: {result['closed_frames']}f",
                                        "",
                                        "=== COUNTER ===",
                                        f"Blinks: {result['blink_count']}",
                                        f"Blinks/min: {result['blink_rate']:.1f}",
                                        f"Yawns: {result['yawn_count']}",
                                        f"Total Yawns: {result['total_yawns']}",
                                        "",
                                        "=== STATUS ===",
                                        f"Yawning State: {'YA' if result['is_yawning_state'] else 'TIDAK'}",
                                        f"Normal EAR: {result['normal_ear']:.3f}",
                                        f"Normal MAR: {result['normal_mar']:.3f}"
                                    ]
                                    
                                    panel_h = len(info_lines) * line_height + 10
                                    if panel_y + panel_h > h - 150:
                                        panel_y = h - panel_h - 150
                                    
                                    # Background panel
                                    cv2.rectangle(frame, (panel_x-5, panel_y-5), 
                                                 (panel_x+200, panel_y+panel_h), 
                                                 (30, 30, 30), -1)
                                    cv2.rectangle(frame, (panel_x-5, panel_y-5), 
                                                 (panel_x+200, panel_y+panel_h), 
                                                 COLOR_INFO, 1)
                                    
                                    # Tampilkan info
                                    for i, text in enumerate(info_lines):
                                        color = COLOR_INFO
                                        if "EAR" in text and "Th" not in text and result['eye_closed']:
                                            color = COLOR_DANGER
                                        elif "MAR" in text and "Th" not in text and result['yawning']:
                                            color = COLOR_DANGER
                                        elif "PITCH" in text and "Th" not in text and "Deviasi" not in text:
                                            color = pitch_color
                                        elif "PERCLOS" in text and result['perclos'] > 15:
                                            color = COLOR_WARN if result['perclos'] < 30 else COLOR_DANGER
                                        elif "Blinks" in text and result['blink_rate'] > 20:
                                            color = COLOR_WARN
                                        elif "Yawns" in text and result['yawn_count'] > 2:
                                            color = COLOR_WARN
                                        elif "Yawning State" in text and result['is_yawning_state']:
                                            color = COLOR_WARN
                                        
                                        cv2.putText(frame, text, (panel_x, panel_y + i*line_height), 
                                                   FONT, FONT_SMALL, color, FONT_THICKNESS)
                                    
                                    # STATUS UTAMA
                                    status_x = w - 220
                                    status_y = 60
                                    
                                    if status_y < 150:
                                        status_y = 150
                                    
                                    # Tentukan status
                                    if result['microsleep']:
                                        status_text = "MICROSLEEP"
                                        status_color = COLOR_DANGER
                                    elif result['score'] >= 70:
                                        status_text = "BAHAYA"
                                        status_color = COLOR_DANGER
                                    elif result['score'] >= 50:
                                        status_text = "WASPADA"
                                        status_color = COLOR_WARN
                                    elif result['score'] >= 30:
                                        status_text = "HATI-HATI"
                                        status_color = (0, 255, 255)
                                    else:
                                        status_text = "AMAN"
                                        status_color = COLOR_SAFE
                                    
                                    # Panel status
                                    cv2.rectangle(frame, (status_x-10, status_y-10), 
                                                 (status_x+210, status_y+45), 
                                                 (30, 30, 30), -1)
                                    cv2.rectangle(frame, (status_x-10, status_y-10), 
                                                 (status_x+210, status_y+45), 
                                                 status_color, 2)
                                    
                                    cv2.putText(frame, "STATUS", 
                                               (status_x, status_y+15), 
                                               FONT, FONT_NORMAL, status_color, FONT_THICKNESS)
                                    cv2.putText(frame, f"{status_text}", 
                                               (status_x, status_y+35), 
                                               FONT, FONT_NORMAL, status_color, FONT_THICKNESS)
                                    
                                    # Skor
                                    cv2.putText(frame, f"Skor: {int(result['score'])}/100", 
                                               (status_x, status_y+55), 
                                               FONT, FONT_SMALL, status_color, FONT_THICKNESS)
                                    
                                    # GRAFIK - PERBAIKAN ERROR ear_range
                                    graph_width = 280
                                    graph_height = 50
                                    graph_spacing = 5
                                    graph_x = w - graph_width - 20
                                    graph_y = h - (graph_height * 3 + graph_spacing * 2 + 40)
                                    
                                    # GRAFIK EAR - PERBAIKAN DI SINI
                                    cv2.rectangle(frame, (graph_x, graph_y), 
                                                 (graph_x + graph_width, graph_y + graph_height), 
                                                 (20, 20, 20), -1)
                                    cv2.rectangle(frame, (graph_x, graph_y), 
                                                 (graph_x + graph_width, graph_y + graph_height), 
                                                 COLOR_INFO, 1)
                                    
                                    # Inisialisasi variabel ear_range di luar blok if
                                    ear_range = 0.3
                                    max_ear = 0.3
                                    min_ear = 0.0
                                    
                                    # Plot EAR history
                                    if len(ear_history) > 1:
                                        points_ear = []
                                        max_ear = max(ear_history) if ear_history else 0.3
                                        min_ear = min(ear_history) if ear_history else 0.0
                                        ear_range = max_ear - min_ear if max_ear > min_ear else 0.3
                                        
                                        for i, val in enumerate(list(ear_history)[-graph_width//2:]):
                                            x = graph_x + i * 2
                                            y = int(graph_y + graph_height - ((val - min_ear) / ear_range * graph_height))
                                            y = max(graph_y, min(y, graph_y + graph_height))
                                            points_ear.append((x, y))
                                        
                                        if len(points_ear) > 1:
                                            cv2.polylines(frame, [np.array(points_ear)], False, COLOR_SAFE, 1)
                                    
                                    # Garis threshold EAR
                                    if ear_range > 0:
                                        ear_th_y = int(graph_y + graph_height - ((THRESH_EAR_CLOSE - min_ear) / ear_range * graph_height))
                                        ear_th_y = max(graph_y, min(ear_th_y, graph_y + graph_height))
                                        cv2.line(frame, (graph_x, ear_th_y), 
                                                 (graph_x + graph_width, ear_th_y), 
                                                 (0, 255, 255), 1)
                                    
                                    cv2.putText(frame, "EAR History", 
                                               (graph_x, graph_y-5), 
                                               FONT, FONT_SMALL, COLOR_SAFE, FONT_THICKNESS)
                                    
                                    # GRAFIK MAR - PERBAIKAN SERUPA
                                    mar_graph_y = graph_y + graph_height + graph_spacing
                                    cv2.rectangle(frame, (graph_x, mar_graph_y), 
                                                 (graph_x + graph_width, mar_graph_y + graph_height), 
                                                 (20, 20, 20), -1)
                                    cv2.rectangle(frame, (graph_x, mar_graph_y), 
                                                 (graph_x + graph_width, mar_graph_y + graph_height), 
                                                 COLOR_INFO, 1)
                                    
                                    # Inisialisasi variabel mar_range di luar blok if
                                    mar_range = 1.0
                                    max_mar = 1.0
                                    min_mar = 0.0
                                    
                                    # Plot MAR history
                                    if len(mar_history) > 1:
                                        points_mar = []
                                        max_mar = max(mar_history) if mar_history else 1.0
                                        min_mar = min(mar_history) if mar_history else 0.0
                                        mar_range = max_mar - min_mar if max_mar > min_mar else 1.0
                                        
                                        for i, val in enumerate(list(mar_history)[-graph_width//2:]):
                                            x = graph_x + i * 2
                                            y = int(mar_graph_y + graph_height - ((val - min_mar) / mar_range * graph_height))
                                            y = max(mar_graph_y, min(y, mar_graph_y + graph_height))
                                            points_mar.append((x, y))
                                        
                                        if len(points_mar) > 1:
                                            cv2.polylines(frame, [np.array(points_mar)], False, (255, 100, 100), 1)
                                    
                                    # Garis threshold MAR
                                    if mar_range > 0:
                                        mar_th_y = int(mar_graph_y + graph_height - ((THRESH_MAR_YAWN - min_mar) / mar_range * graph_height))
                                        mar_th_y = max(mar_graph_y, min(mar_th_y, mar_graph_y + graph_height))
                                        cv2.line(frame, (graph_x, mar_th_y), 
                                                 (graph_x + graph_width, mar_th_y), 
                                                 (0, 255, 255), 1)
                                    
                                    cv2.putText(frame, "MAR History", 
                                               (graph_x, mar_graph_y-5), 
                                               FONT, FONT_SMALL, (255, 100, 100), FONT_THICKNESS)
                                    
                                    # GRAFIK PITCH
                                    pitch_graph_y = mar_graph_y + graph_height + graph_spacing
                                    cv2.rectangle(frame, (graph_x, pitch_graph_y), 
                                                 (graph_x + graph_width, pitch_graph_y + graph_height), 
                                                 (20, 20, 20), -1)
                                    cv2.rectangle(frame, (graph_x, pitch_graph_y), 
                                                 (graph_x + graph_width, pitch_graph_y + graph_height), 
                                                 COLOR_INFO, 1)
                                    
                                    # Plot PITCH history
                                    if len(pitch_history) > 1:
                                        points_pitch = []
                                        min_pitch = -45
                                        max_pitch = 45
                                        pitch_range = max_pitch - min_pitch
                                        
                                        for i, val in enumerate(list(pitch_history)[-graph_width//2:]):
                                            x = graph_x + i * 2
                                            y = int(pitch_graph_y + graph_height - ((val - min_pitch) / pitch_range * graph_height))
                                            y = max(pitch_graph_y, min(y, pitch_graph_y + graph_height))
                                            points_pitch.append((x, y))
                                        
                                        if len(points_pitch) > 1:
                                            cv2.polylines(frame, [np.array(points_pitch)], False, (255, 150, 50), 1)
                                    
                                    # Garis baseline pitch
                                    min_pitch = -45
                                    max_pitch = 45
                                    pitch_range = max_pitch - min_pitch
                                    
                                    if pitch_range > 0:
                                        baseline_y = int(pitch_graph_y + graph_height - ((baseline_values['pitch'] - min_pitch) / pitch_range * graph_height))
                                        baseline_y = max(pitch_graph_y, min(baseline_y, pitch_graph_y + graph_height))
                                        cv2.line(frame, (graph_x, baseline_y), 
                                                 (graph_x + graph_width, baseline_y), 
                                                 (0, 255, 0), 1, cv2.LINE_AA)
                                    
                                    # Garis threshold bawah
                                    if pitch_range > 0:
                                        pitch_th_down_y = int(pitch_graph_y + graph_height - ((pitch_th_down - min_pitch) / pitch_range * graph_height))
                                        pitch_th_down_y = max(pitch_graph_y, min(pitch_th_down_y, pitch_graph_y + graph_height))
                                        cv2.line(frame, (graph_x, pitch_th_down_y), 
                                                 (graph_x + graph_width, pitch_th_down_y), 
                                                 (0, 0, 255), 1, cv2.LINE_AA)
                                    
                                    # Garis threshold atas
                                    if pitch_range > 0:
                                        pitch_th_up_y = int(pitch_graph_y + graph_height - ((pitch_th_up - min_pitch) / pitch_range * graph_height))
                                        pitch_th_up_y = max(pitch_graph_y, min(pitch_th_up_y, pitch_graph_y + graph_height))
                                        cv2.line(frame, (graph_x, pitch_th_up_y), 
                                                 (graph_x + graph_width, pitch_th_up_y), 
                                                 (255, 165, 0), 1, cv2.LINE_AA)
                                    
                                    # Garis nol (0 derajat)
                                    zero_y = int(pitch_graph_y + graph_height - ((0 - min_pitch) / pitch_range * graph_height))
                                    zero_y = max(pitch_graph_y, min(zero_y, pitch_graph_y + graph_height))
                                    cv2.line(frame, (graph_x, zero_y), 
                                             (graph_x + graph_width, zero_y), 
                                             (200, 200, 200), 1, cv2.LINE_AA)
                                    
                                    cv2.putText(frame, "PITCH History", 
                                               (graph_x, pitch_graph_y-5), 
                                               FONT, FONT_SMALL, (255, 150, 50), FONT_THICKNESS)
                                    cv2.putText(frame, f"Baseline: {baseline_values['pitch']:.1f} deg", 
                                               (graph_x + graph_width - 100, pitch_graph_y-5), 
                                               FONT, 0.3, (0, 255, 0), 1)
                                    
                                    # ALERT VISUAL DAN AUDIO
                                    current_time_real = time.time()
                                    
                                    if result['microsleep']:
                                        # Flash effect
                                        if int(current_time_real * 2) % 2 == 0:
                                            overlay = frame.copy()
                                            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                                            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                                        
                                        # Text warning
                                        warning_text = "!!! MICROSLEEP DETECTED !!!"
                                        text_size = cv2.getTextSize(warning_text, FONT, 0.7, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, warning_text, 
                                                   (text_x, h//2), 
                                                   FONT, 0.7, COLOR_DANGER, 2)
                                        
                                        # Play microsleep alert sound
                                        audio_manager.play_audio('microsleep', sound_microsleep)
                                    
                                    elif result['score'] >= 70 and not result['microsleep']:
                                        # Flash effect for danger
                                        if int(current_time_real * 2) % 2 == 0:
                                            overlay = frame.copy()
                                            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                                            cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
                                        
                                        warning_text = "!!! PERINGATAN KELELAHAN !!!"
                                        text_size = cv2.getTextSize(warning_text, FONT, 0.7, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, warning_text, 
                                                   (text_x, h//2), 
                                                   FONT, 0.7, COLOR_DANGER, 2)
                                        
                                        # Play warning sound
                                        audio_manager.play_audio('warning', sound_warning)
                                    
                                    elif result['score'] >= 50 and not result['microsleep']:
                                        # Visual warning for moderate fatigue
                                        warning_text = "WASPADA - Tanda Kelelahan"
                                        text_size = cv2.getTextSize(warning_text, FONT, 0.6, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, warning_text, 
                                                   (text_x, h//2 - 50), 
                                                   FONT, 0.6, COLOR_WARN, 2)
                                        
                                        # Play warning sound
                                        audio_manager.play_audio('warning', sound_warning)
                                    
                                    # Tampilkan info yawning jika sedang terjadi
                                    if result['yawning'] or result['is_yawning_state']:
                                        yawn_text = "MENGUAP TERDETEKSI"
                                        text_size = cv2.getTextSize(yawn_text, FONT, 0.6, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, yawn_text, 
                                                   (text_x, h//2 + 30), 
                                                   FONT, 0.6, (0, 200, 255), 2)
                                    
                                    # Tampilkan info head down/up
                                    if result['head_down_raw']:
                                        head_text = "KEPALA MENUNDUK"
                                        text_size = cv2.getTextSize(head_text, FONT, 0.6, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, head_text, 
                                                   (text_x, h//2 + 60), 
                                                   FONT, 0.6, COLOR_DANGER, 2)
                                    elif result['head_up']:
                                        head_text = "KEPALA MENENGADAH"
                                        text_size = cv2.getTextSize(head_text, FONT, 0.6, 2)[0]
                                        text_x = w // 2 - text_size[0] // 2
                                        cv2.putText(frame, head_text, 
                                                   (text_x, h//2 + 60), 
                                                   FONT, 0.6, COLOR_WARN, 2)
            
            # HEADER
            cv2.rectangle(frame, (0, 0), (w, 35), (40, 40, 40), -1)
            status_text = "KALIBRASI" if not calibrated else "MONITORING"
            header_text = f"DROWSINESS DETECTION | Video: {video_time:.1f}/{duration:.1f}s | Status: {status_text} | FPS: {int(fps)}"
            cv2.putText(frame, header_text, (10, 22), FONT, FONT_SMALL, COLOR_INFO, 1)
            
            # FOOTER
            cv2.rectangle(frame, (0, h-22), (w, h), (40, 40, 40), -1)
            cv2.putText(frame, "q:Keluar | s:Screenshot | p:Pause/Play", 
                       (10, h-5), FONT, FONT_SMALL, COLOR_INFO, 1)
            
            # TAMPILKAN FRAME
            cv2.imshow(f'Deteksi Kelelahan - Video: {video_path}', frame)
            
            # Handle keyboard
            key = cv2.waitKey(int(1000 / fps_video)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"drowsiness_video_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot disimpan: {filename}")
            elif key == ord('p'):
                # Pause/Play
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('p'):
                        break
                    elif key2 == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("[INFO] Membersihkan resource...")
        cap.release()
        cv2.destroyAllWindows()
        if audio_available:
            pygame.mixer.quit()
        
        # Export CSV
        print("\n[INFO] Menyimpan hasil ke CSV...")
        event_recorder.save_csv(Path(video_path).parent, current_frame)

# ============================================
# JALANKAN PROGRAM
# ============================================
if __name__ == "__main__":
    main()