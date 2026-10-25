import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import math
import subprocess
import traceback
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

# ============================================
# KONFIGURASI SISTEM
# ============================================
W_EYE = 0.45
W_MOUTH = 0.25
W_HEAD = 0.30

THRESH_EAR_CLOSE = 0.18
THRESH_EAR_BLINK = 0.22
THRESH_MAR_YAWN = 0.20
THRESH_HEAD_PITCH = 10.0

LEVEL_HATI_HATI = 20
LEVEL_WASPADA = 40
LEVEL_BAHAYA = 60

TIME_MICROSLEEP_LIMIT = 1.5
TIME_YAWN_MIN_DURATION = 1.5
TIME_BLINK_MIN_INTERVAL = 0.2
TIME_CONSECUTIVE_FRAMES = 15
TIME_PERCLOS_WINDOW = 60

USE_OSCILLATION_GUARD = True
USE_PEAK_GATE = True
USE_PERCLOS_WARMUP = True

CAM_WIDTH = 640
CAM_HEIGHT = 480

COLOR_SAFE = (0, 255, 0)
COLOR_WARN = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_INFO = (200, 200, 200)
COLOR_DEBUG = (255, 255, 0)
COLOR_ROI = (0, 255, 255)

AUDIO_DIR = Path(__file__).resolve().parent / "audio"


class RPiCameraFeed:
    def __init__(self, tuning_file="/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json", width=CAM_WIDTH, height=CAM_HEIGHT, fps=30):
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
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.use_rpicam:
            raw_data = self.process.stdout.read(self.frame_size)
            if not raw_data or len(raw_data) != self.frame_size:
                return False, None
            yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        return self.cap.read()

    def set(self, prop_id, value):
        if self.use_rpicam:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                self.width = int(value)
            elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                self.height = int(value)
            return True
        return self.cap.set(prop_id, value) if self.cap else False

    def get(self, prop_id):
        if self.use_rpicam:
            if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.width)
            if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.height)
            return 0.0
        return self.cap.get(prop_id) if self.cap else 0.0

    def release(self):
        if self.use_rpicam and self.process:
            self.process.terminate()
            self.process.wait(timeout=2)
        elif self.cap:
            self.cap.release()


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_landmark_point(landmark, index, width, height):
    return (int(landmark[index].x * width), int(landmark[index].y * height))


def calculate_ear_single_eye(eye_points):
    a = euclidean_distance(eye_points[1], eye_points[5])
    b = euclidean_distance(eye_points[2], eye_points[4])
    c = euclidean_distance(eye_points[0], eye_points[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


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
    left_corner = get_landmark_point(landmarks, 61, width, height)
    right_corner = get_landmark_point(landmarks, 291, width, height)
    upper_left = get_landmark_point(landmarks, 82, width, height)
    lower_left = get_landmark_point(landmarks, 87, width, height)
    upper_mid = get_landmark_point(landmarks, 13, width, height)
    lower_mid = get_landmark_point(landmarks, 14, width, height)
    upper_right = get_landmark_point(landmarks, 312, width, height)
    lower_right = get_landmark_point(landmarks, 317, width, height)

    a = euclidean_distance(upper_left, lower_left)
    b = euclidean_distance(upper_mid, lower_mid)
    c = euclidean_distance(upper_right, lower_right)
    d = euclidean_distance(left_corner, right_corner)
    if d == 0:
        return 0.0
    return (a + b + c) / (2.0 * d)


def calculate_head_pose_improved(landmarks, width, height):
    try:
        forehead = get_landmark_point(landmarks, 10, width, height)
        chin = get_landmark_point(landmarks, 152, width, height)
        nose_tip = get_landmark_point(landmarks, 4, width, height)

        ax = chin[0] - forehead[0]
        ay = chin[1] - forehead[1]
        face_len_sq = ax * ax + ay * ay
        if face_len_sq < 1.0:
            return 0.0

        nx = nose_tip[0] - forehead[0]
        ny = nose_tip[1] - forehead[1]
        t = (nx * ax + ny * ay) / face_len_sq
        pitch = (0.45 - t) * 120.0
        return max(-45.0, min(45.0, pitch))
    except Exception:
        return 0.0


class PERCLOSCalculator:
    def __init__(self, ws=60, fps=30):
        self.ws = int(ws * fps)
        self.es = deque(maxlen=self.ws)
        self.bc = 0
        self.lbt = time.time()
        self.br = 0
        self.les = False

    def update(self, ear, th):
        c = ear < th
        if self.les and not c:
            self.bc += 1
        self.les = c
        self.es.append(c)
        p = (sum(self.es) / len(self.es) * 100.0) if len(self.es) >= self.ws // 2 else 0.0
        t = time.time()
        if (t - self.lbt) >= 60:
            self.br = self.bc
            self.bc = 0
            self.lbt = t
        return {
            'perclos': p,
            'blink_count': self.bc,
            'blink_rate': self.br,
            'window_frames': len(self.es),
            'closed_frames': int(sum(self.es))
        }


class DrowsinessAccumulator:
    def __init__(self):
        self.mouth_acc = 0.0
        self.head_acc = 0.0
        self.eye_acc = 0.0
        self.microsleep_acc = 0.0

        self.mouth_decay_per_sec = 0.25
        self.head_decay_per_sec = 0.30
        self.eye_decay_per_sec = 0.20
        self.microsleep_decay_per_sec = 0.35

        self.mouth_event_gain = 24.0
        self.head_event_gain = 22.0
        self.microsleep_event_gain = 45.0
        self.head_hold_gain_per_sec = 2.2
        self.eye_hold_gain_per_sec = 1.8
        self.perclos_gain_per_sec = 0.7
        self.microsleep_hold_gain_per_sec = 6.0

        self.recent_window_sec = 300.0
        self.recent_yawns = deque()
        self.recent_head_drops = deque()
        self.recent_microsleeps = deque()
        self.microsleep_count = 0

    def update(self, dt, now_ts, head_down_raw, eye_closed_raw, perclos_value, microsleep_detected):
        dt = max(0.0, float(dt))
        decay_scale = 0.55 if (head_down_raw or eye_closed_raw or perclos_value > 20.0 or microsleep_detected) else 1.0

        self.mouth_acc = max(0.0, self.mouth_acc - self.mouth_decay_per_sec * dt)
        self.head_acc = max(0.0, self.head_acc - self.head_decay_per_sec * dt * decay_scale)
        self.eye_acc = max(0.0, self.eye_acc - self.eye_decay_per_sec * dt * decay_scale)
        self.microsleep_acc = max(0.0, self.microsleep_acc - self.microsleep_decay_per_sec * dt * decay_scale)

        if head_down_raw:
            self.head_acc = min(100.0, self.head_acc + self.head_hold_gain_per_sec * dt)
        if eye_closed_raw:
            self.eye_acc = min(100.0, self.eye_acc + self.eye_hold_gain_per_sec * dt)
        if perclos_value > 20.0:
            self.eye_acc = min(100.0, self.eye_acc + self.perclos_gain_per_sec * dt * (perclos_value / 20.0))
        if microsleep_detected:
            self.microsleep_acc = min(100.0, self.microsleep_acc + self.microsleep_hold_gain_per_sec * dt)

        for d in [self.recent_yawns, self.recent_head_drops, self.recent_microsleeps]:
            while d and (now_ts - d[0]) > self.recent_window_sec:
                d.popleft()

    def add_yawn(self, ts):
        self.mouth_acc = min(100.0, self.mouth_acc + self.mouth_event_gain)
        self.recent_yawns.append(ts)

    def add_head_drop(self, ts):
        self.head_acc = min(100.0, self.head_acc + self.head_event_gain)
        self.recent_head_drops.append(ts)

    def add_microsleep(self, ts):
        self.microsleep_acc = min(100.0, self.microsleep_acc + self.microsleep_event_gain)
        self.recent_microsleeps.append(ts)
        self.microsleep_count += 1

    def get_eye_score(self):
        return min(100.0, self.eye_acc + min(45.0, self.microsleep_acc * 0.55 + len(self.recent_microsleeps) * 12.0))

    def get_mouth_score(self):
        return min(100.0, self.mouth_acc + min(40.0, len(self.recent_yawns) * 8.0))

    def get_head_score(self):
        return min(100.0, self.head_acc + min(40.0, len(self.recent_head_drops) * 10.0))


class DrowsinessDetector:
    # Blink/yawn/headdrop/microsleep event logic adapted from fast_eval_cache.py.
    # Decay/accumulator/scoring logic aligned with v5_pengujian1dan2_rpi5.py.
    def __init__(
        self,
        fps=30,
        baseline_pitch=0.0,
        baseline_ear=0.30,
        baseline_mouth=0.05,
        th_ear_close=0.18,
        th_ear_open=0.20,
        th_mar_yawn=0.35
    ):
        self.fps = fps
        self.bp = baseline_pitch
        self.be = baseline_ear
        self.bm = baseline_mouth
        self.th_ear_close = th_ear_close
        self.th_ear_open = th_ear_open
        self.th_mar_yawn = th_mar_yawn

        self.eb = deque(maxlen=3)
        self.pb = deque(maxlen=17)

        self.ecf = 0
        self.ecf_grace = 0
        self.ecf_blink = 0
        self.vbc = 0
        self.blink_active = False
        self.last_blink_ts = 0.0
        self.blink_rearm_ts = 0.0
        self.blink_rearm_ready = True

        self.ys = 'IDLE'
        self.yawn_start_t = 0.0
        self.yhf = 0
        self.ypm = 0.0
        self.lyt = 0.0
        self.max_p_in_yawn = 0.0
        self.min_e_in_yawn = 1.0
        self.p_sum = 0.0
        self.p_count = 0

        self.hd_grace_end = 0.0
        self.hd_start_p2 = None
        self.hd_grace_end_p2 = 0.0
        self.head_down_start = None

        self.ms_confirmed = False
        self.hd_confirmed = False

        self.pitch_threshold_high = 15.0
        self.pitch_scale_close = 0.75
        self.pitch_scale_open = 0.85

        self.perclos_calc = PERCLOSCalculator(TIME_PERCLOS_WINDOW, fps)
        self.acc = DrowsinessAccumulator()

        self.last_update_ts = time.time()
        self.pm = False
        self.ph = False

    def update(self, ear, mar, pitch, fps=30):
        t = time.time()
        events = []

        self.eb.append(ear)
        es = float(np.median(self.eb)) if len(self.eb) >= 3 else ear

        self.pb.append(pitch if pitch is not None else self.bp)
        ps = float(np.median(list(self.pb)))
        pd_val = ps - self.bp

        if pd_val > self.pitch_threshold_high:
            adapt_factor_close = self.pitch_scale_close
            adapt_factor_open = self.pitch_scale_open
        else:
            adapt_factor_close = 1.0
            adapt_factor_open = 1.0

        th_ear_close_adapted = self.th_ear_close * adapt_factor_close
        th_ear_open_adapted = self.th_ear_open * adapt_factor_open

        ec = (es < th_ear_close_adapted) and not (pd_val < -25 and es > th_ear_open_adapted * 0.5)
        if ec:
            self.ecf_blink += 1
            if not self.blink_active and self.ecf_blink >= 2:
                if t >= self.blink_rearm_ts and (t - self.last_blink_ts) >= 0.5:
                    events.append('blink')
                    self.vbc += 1
                    self.last_blink_ts = t
                    self.blink_rearm_ts = t + 0.6
                self.blink_active = True
        elif es > th_ear_open_adapted:
            self.ecf_blink = 0
            self.blink_active = False
            if es > (self.be * 0.90):
                self.blink_rearm_ready = True

        mar_th = 0.40
        bl_p = self.bp
        if self.ys == 'IDLE' and mar >= mar_th:
            self.ys = 'BUILDING'
            self.yawn_start_t = t
            self.yhf = 1
            self.ypm = mar
            self.max_p_in_yawn = pitch
            self.min_e_in_yawn = ear
            self.p_sum = pitch - bl_p
            self.p_count = 1
        elif self.ys == 'BUILDING':
            if mar < mar_th:
                self.ys = 'IDLE'
                self.yhf = 0
            else:
                self.yhf += 1
                self.ypm = max(self.ypm, mar)
                self.max_p_in_yawn = max(self.max_p_in_yawn, pitch)
                self.min_e_in_yawn = min(self.min_e_in_yawn, ear)
                self.p_sum += (pitch - bl_p)
                self.p_count += 1

                dur = t - self.yawn_start_t
                sq = (self.min_e_in_yawn < (self.be * 0.95))
                ear_rel = self.min_e_in_yawn / max(self.be, 1e-6)
                avg_p = self.p_sum / max(self.p_count, 1)

                is_yawn = False
                if dur >= 1.2 and self.ypm >= 0.60 and avg_p > 1.8:
                    is_yawn = True
                if dur >= 1.2 and self.ypm >= 0.90 and avg_p > 3.5:
                    is_yawn = True
                if dur >= 1.2 and self.ypm >= 0.42 and avg_p > 5.0 and sq:
                    is_yawn = True
                if dur >= 1.0 and self.ypm >= 0.45 and ear_rel < 0.50 and avg_p > 0.1:
                    is_yawn = True

                if is_yawn and (self.min_e_in_yawn < self.be * 0.99 or self.ypm > 0.80):
                    self.ys = 'CONFIRMED'
                    self.lyt = t
                    events.append('yawn')
        elif self.ys == 'CONFIRMED' and mar < mar_th:
            self.ys = 'COOLDOWN'
        elif self.ys == 'COOLDOWN' and (t - self.lyt) >= 4.0:
            self.ys = 'IDLE'

        ec_micro = es < (self.th_ear_close * 0.50)
        is_yawning = self.ys in ('BUILDING', 'CONFIRMED')
        if ec_micro and not is_yawning and mar <= (self.bm + 0.05):
            self.ecf += 1
            self.ecf_grace = 0
        elif self.ecf > 0 and self.ecf_grace < 4:
            self.ecf_grace += 1
        else:
            self.ecf = 0
            self.ecf_grace = 0

        m_det = False
        if self.ecf >= 30:
            if not self.ms_confirmed:
                events.append('microsleep')
                self.ms_confirmed = True
            m_det = True
        else:
            if es > self.th_ear_open:
                self.ms_confirmed = False

        ear_reliable = ear is not None and ear >= 0.05
        is_below_p1 = ear_reliable and (pd_val < -11.0)
        is_below_p2 = ear_reliable and (pd_val < -25.0)

        if is_below_p1:
            self.hd_grace_end = t + 0.2
            if self.head_down_start is None:
                self.head_down_start = t
        else:
            if t > self.hd_grace_end:
                self.head_down_start = None

        if is_below_p2:
            self.hd_grace_end_p2 = t + 0.1
            if self.hd_start_p2 is None:
                self.hd_start_p2 = t
        else:
            if t > self.hd_grace_end_p2:
                self.hd_start_p2 = None

        hd_confirmed_now = False
        p1_triggered = (self.head_down_start is not None and (t - self.head_down_start) >= 0.6)
        p2_triggered = (self.hd_start_p2 is not None and (t - self.hd_start_p2) >= 0.3)
        if p1_triggered or p2_triggered:
            if not self.hd_confirmed:
                events.append('headdrop')
                self.hd_confirmed = True
            hd_confirmed_now = True
        if (not hd_confirmed_now and not is_below_p1 and not is_below_p2 and
                t > self.hd_grace_end and t > self.hd_grace_end_p2):
            self.hd_confirmed = False

        hd_det = self.hd_confirmed
        head_down_raw = is_below_p1 or is_below_p2
        head_up = pd_val > THRESH_HEAD_PITCH and ps > 3.0

        perclos_data = self.perclos_calc.update(es, THRESH_EAR_CLOSE)

        dt = min(t - self.last_update_ts, 0.5)
        self.last_update_ts = t

        if 'microsleep' in events and not self.pm:
            self.acc.add_microsleep(t)
        if 'headdrop' in events and not self.ph:
            self.acc.add_head_drop(t)
        if 'yawn' in events:
            self.acc.add_yawn(t)

        self.pm = m_det
        self.ph = hd_det

        self.acc.update(dt, t, head_down_raw, ec, perclos_data['perclos'], m_det)

        instant_score = 0.0
        if m_det:
            instant_score += W_EYE * 100.0
        elif self.ecf >= 15:
            instant_score += W_EYE * 60.0
        elif perclos_data['perclos'] > 20.0:
            instant_score += W_EYE * (min(perclos_data['perclos'], 40.0) / 40.0 * 70.0)

        if self.ys == 'CONFIRMED':
            instant_score += W_MOUTH * 70.0

        if hd_det:
            instant_score += W_HEAD * 100.0
        elif head_up:
            instant_score += W_HEAD * 30.0

        eye_s = self.acc.get_eye_score()
        mouth_s = self.acc.get_mouth_score()
        head_s = self.acc.get_head_score()

        severity_bonus = 0.15 * max(eye_s, mouth_s, head_s)
        total_events = len(self.acc.recent_microsleeps) + len(self.acc.recent_yawns) + len(self.acc.recent_head_drops)
        if total_events > 1:
            severity_bonus += (total_events - 1) * 6.0

        accumulator_score = min(100.0, W_EYE * eye_s + W_MOUTH * mouth_s + W_HEAD * head_s + severity_bonus)
        score = max(0.0, min(100.0, 0.35 * instant_score + 0.65 * accumulator_score))

        return {
            'ear': es,
            'mar': mar,
            'pitch': ps,
            'pitch_deviation': pd_val,
            'perclos': perclos_data['perclos'],
            'blink_rate': perclos_data['blink_rate'],
            'eye_closed': ec,
            'microsleep': m_det,
            'yawning': self.ys == 'CONFIRMED',
            'head_down': hd_det,
            'head_down_raw': head_down_raw,
            'head_up': head_up,
            'score': score,
            'instant_score': instant_score,
            'accumulator_score': accumulator_score,
            'inst': instant_score,
            'acc': accumulator_score,
            'blink_count': self.vbc,
            'yawn_count': len(self.acc.recent_yawns),
            'headdrop_count': len(self.acc.recent_head_drops),
            'total_yawns': len(self.acc.recent_yawns),
            'valid_blink_count': self.vbc,
            'window_frames': perclos_data['window_frames'],
            'closed_frames': perclos_data['closed_frames'],
            'normal_ear': self.be,
            'normal_mar': self.bm,
            'yawn_peak_mar': self.ypm,
            'yawn_state': self.ys,
            'yawn_hold_frames': self.yhf,
            'is_talking': False,
            'mouth_acc': self.acc.mouth_acc,
            'head_acc': self.acc.head_acc,
            'eye_acc': self.acc.eye_acc,
            'microsleep_acc': self.acc.microsleep_acc,
            'recent_yawns_10m': len(self.acc.recent_yawns),
            'recent_head_drops_10m': len(self.acc.recent_head_drops),
            'recent_microsleeps_10m': len(self.acc.recent_microsleeps),
            'microsleep_count': self.acc.microsleep_count
        }


class AudioManager:
    FACE_MISSING_DELAY_SEC = 2.0
    FACE_MISSING_COOLDOWN_SEC = 15.0
    MICROSLEEP_COOLDOWN_SEC = 20.0
    LEVEL_COOLDOWN_SEC = 180.0

    SOUND_FILES = {
        'startup': 'vo_01_init_dan_instruksi.wav',
        'calibration_start': 'vo_02_mulai_kalibrasi.wav',
        'calibration_success': 'vo_03_kalibrasi_berhasil.wav',
        'face_missing': 'vo_04_wajah_tidak_terdeteksi.wav',
        'alarm': 'alarm.mp3',
        'microsleep_voice': 'vo_05_alert_microsleep.wav',
        'level_caution': 'vo_06_level_hati_hati.wav',
        'level_warning': 'vo_07_level_waspada.wav',
        'level_danger': 'vo_08_level_bahaya.wav',
    }

    LEVEL_SOUNDS = {
        1: 'level_caution',
        2: 'level_warning',
        3: 'level_danger',
    }

    def __init__(self, audio_dir=AUDIO_DIR, channel_id=0):
        self.enabled = False
        self.channel = None
        self.sounds = {}
        self.current_tag = None
        self.init_tag = 'startup'
        self.microsleep_sequence_active = False
        self.microsleep_cooldown_until = 0.0
        self.face_missing_since = None
        self.face_missing_cooldown_until = 0.0
        self.level_last_played = {1: 0.0, 2: 0.0, 3: 0.0}
        self.last_announced_level = 0

        try:
            if pygame is None:
                raise RuntimeError("pygame belum terinstal")
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(max(4, pygame.mixer.get_num_channels()))
            self.channel = pygame.mixer.Channel(channel_id)
            self.sounds = {
                tag: pygame.mixer.Sound(str(Path(audio_dir) / filename))
                for tag, filename in self.SOUND_FILES.items()
            }
            self.enabled = True
            print("[INFO] AudioManager aktif.")
        except Exception as exc:
            print(f"[WARNING] AudioManager nonaktif: {exc}")

    def shutdown(self):
        if self.enabled:
            self.channel.stop()
            pygame.mixer.quit()

    def update(self):
        if not self.enabled:
            return
        if self.microsleep_sequence_active and not self.channel.get_busy():
            self.microsleep_sequence_active = False
            self.current_tag = None
            self.microsleep_cooldown_until = time.time() + self.MICROSLEEP_COOLDOWN_SEC
        elif self.current_tag and not self.channel.get_busy():
            self.current_tag = None

    def is_init_playing(self):
        return self.enabled and self.current_tag == self.init_tag and self.channel.get_busy()

    def is_busy(self):
        return self.enabled and self.channel.get_busy()

    def _play(self, tag, force=False):
        if not self.enabled:
            return False
        if self.is_init_playing() and tag != self.init_tag:
            return False
        if self.channel.get_busy():
            if not force:
                return False
            self.channel.stop()
        self.channel.play(self.sounds[tag])
        self.current_tag = tag
        return True

    def play_startup(self):
        self._play(self.init_tag, force=True)

    def play_calibration_start(self):
        if not self.enabled:
            return True
        if self.is_init_playing():
            return False
        self.microsleep_sequence_active = False
        self.current_tag = None
        return self._play('calibration_start', force=True)

    def play_calibration_success(self):
        self._play('calibration_success', force=True)

    def reset_for_calibration(self):
        self.face_missing_since = None
        self.face_missing_cooldown_until = 0.0
        self.microsleep_sequence_active = False
        self.microsleep_cooldown_until = 0.0
        self.last_announced_level = 0
        self.play_startup()

    def update_face_presence(self, has_face, state):
        if not self.enabled:
            return
        now = time.time()
        if has_face:
            self.face_missing_since = None
            if self.current_tag == 'face_missing' and self.channel.get_busy():
                self.channel.stop()
                self.current_tag = None
            return

        if state != "MONITORING" or self.is_init_playing():
            self.face_missing_since = None
            return

        if self.face_missing_since is None:
            self.face_missing_since = now
            return

        if (now - self.face_missing_since >= self.FACE_MISSING_DELAY_SEC and
                now >= self.face_missing_cooldown_until and
                not self.microsleep_sequence_active and
                not self.channel.get_busy()):
            if self._play('face_missing'):
                self.face_missing_cooldown_until = now + self.FACE_MISSING_COOLDOWN_SEC

    def handle_monitoring_result(self, result):
        if not self.enabled or self.is_init_playing():
            return

        self.update()
        if result.get('microsleep'):
            self._trigger_microsleep_sequence()
            return

        if self.microsleep_sequence_active or self.channel.get_busy():
            return

        self._trigger_level_warning(result.get('score', 0.0))

    def _trigger_microsleep_sequence(self):
        now = time.time()
        if self.microsleep_sequence_active or now < self.microsleep_cooldown_until:
            return
        self.channel.stop()
        self.channel.play(self.sounds['alarm'])
        self.channel.queue(self.sounds['microsleep_voice'])
        self.current_tag = 'microsleep_sequence'
        self.microsleep_sequence_active = True

    def _score_to_level(self, score):
        if score >= LEVEL_BAHAYA:
            return 3
        if score >= LEVEL_WASPADA:
            return 2
        if score >= LEVEL_HATI_HATI:
            return 1
        return 0

    def _trigger_level_warning(self, score):
        now = time.time()
        level = self._score_to_level(score)
        if level == 0:
            self.last_announced_level = 0
            return

        escalated = level > self.last_announced_level
        cooldown_ready = (now - self.level_last_played[level]) >= self.LEVEL_COOLDOWN_SEC
        if escalated or cooldown_ready:
            if self._play(self.LEVEL_SOUNDS[level]):
                self.level_last_played[level] = now
                self.last_announced_level = max(self.last_announced_level, level)


def main():
    global THRESH_EAR_CLOSE, THRESH_MAR_YAWN, THRESH_HEAD_PITCH, THRESH_EAR_BLINK

    print("[INFO] Memulai MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.8)

    print("[INFO] Membuka Kamera...")
    tuning_file = "/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json"
    cap = RPiCameraFeed(tuning_file=tuning_file, width=CAM_WIDTH, height=CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

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

    audio_manager = AudioManager()
    audio_manager.play_startup()

    detector = None
    state = "CALIBRATION"
    calibration_data = []
    baseline_values = {}

    fps_ema = 30.0
    prev_frame_time = time.time()
    calibration_start_time = None
    show_graphs = True
    ui_compact_mode = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_small = 0.55
    font_normal = 0.65
    font_thickness = 1

    ear_history = deque(maxlen=100)
    mar_history = deque(maxlen=100)
    pitch_history = deque(maxlen=100)

    def draw_text_with_outline(img, text, pos, fnt, scale, color, thickness):
        cv2.putText(img, text, pos, fnt, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos, fnt, scale, color, thickness, cv2.LINE_AA)

    print("\n[INFO] Sistem siap. Tekan:")
    print("  'c' - Mulai Kalibrasi (3 detik)")
    print("  'q' - Keluar")
    print("  'r' - Reset")
    print("  's' - Simpan Screenshot")
    print("  'h' - Toggle Graph")
    print("  'u' - Toggle UI Compact")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[FATAL] Koneksi ke kamera terputus! (cap.read() = False)")
                break

            frame_original = frame.copy()

            curr_frame_time = time.time()
            inst_fps = 1.0 / (curr_frame_time - prev_frame_time + 1e-6)
            prev_frame_time = curr_frame_time
            fps_ema = 0.9 * fps_ema + 0.1 * inst_fps
            fps = fps_ema

            roi_frame = None

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_mesh = face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            has_face = bool(results_mesh.multi_face_landmarks)
            audio_manager.update()
            audio_manager.update_face_presence(has_face, state)

            if has_face:
                ih, iw, _ = frame.shape
                for face_landmarks in results_mesh.multi_face_landmarks:
                    x_c = [lm.x for lm in face_landmarks.landmark]
                    y_c = [lm.y for lm in face_landmarks.landmark]
                    x_min, x_max = min(x_c), max(x_c)
                    y_min, y_max = min(y_c), max(y_c)
                    x = int(x_min * iw)
                    y = int(y_min * ih)
                    w_bbox = int((x_max - x_min) * iw)
                    h_bbox = int((y_max - y_min) * ih)
                    padding = 50
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(iw, x + w_bbox + padding)
                    y2 = min(ih, y + h_bbox + padding)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)
                    cv2.putText(frame, "FaceMesh ROI", (x1, y1 - 10), font, font_small, COLOR_ROI, font_thickness)

                    roi_frame = frame_original[y1:y2, x1:x2]

                    for i in [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]:
                        cv2.circle(frame, get_landmark_point(face_landmarks.landmark, i, iw, ih), 2, (0, 255, 0), -1)
                    for i in [13, 14, 78, 308]:
                        cv2.circle(frame, get_landmark_point(face_landmarks.landmark, i, iw, ih), 2, (255, 0, 0), -1)

                    p10 = get_landmark_point(face_landmarks.landmark, 10, iw, ih)
                    p152 = get_landmark_point(face_landmarks.landmark, 152, iw, ih)
                    cv2.circle(frame, p10, 3, (255, 200, 0), -1)
                    cv2.circle(frame, p152, 3, (255, 0, 255), -1)
                    cv2.line(frame, p10, p152, (255, 255, 0), 1)

                    ear, _, _ = calculate_ear(face_landmarks.landmark, iw, ih)
                    mar = calculate_mar(face_landmarks.landmark, iw, ih)
                    pitch = calculate_head_pose_improved(face_landmarks.landmark, iw, ih)

                    ear_history.append(ear)
                    mar_history.append(mar)
                    pitch_history.append(pitch)

                    if state == "CALIBRATION":
                        cv2.rectangle(frame, (w // 2 - 250, h // 2 - 130), (w // 2 + 250, h // 2 + 200), (30, 30, 30), -1)
                        cv2.rectangle(frame, (w // 2 - 250, h // 2 - 130), (w // 2 + 250, h // 2 + 200), COLOR_WARN, 2)
                        cv2.putText(frame, "KALIBRASI WAJAH", (w // 2 - 80, h // 2 - 100), font, font_normal, COLOR_WARN, 2, cv2.LINE_AA)

                        instructions = [
                            "1. Hadap kamera dengan posisi normal",
                            "2. Mata terbuka normal",
                            "3. Mulut tertutup",
                            "4. Kepala lurus ke depan",
                            "5. Tekan 'C' untuk mulai (3 detik)",
                            "6. Tunggu hingga selesai"
                        ]
                        for i, text in enumerate(instructions):
                            cv2.putText(frame, text, (w // 2 - 230, h // 2 - 60 + i * 22), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)

                        rt_y = h // 2 + 90
                        cv2.putText(frame, "=== NILAI REAL-TIME ===", (w // 2 - 230, rt_y), font, font_small, COLOR_DEBUG, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"EAR: {ear:.3f}", (w // 2 - 230, rt_y + 20), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"MAR: {mar:.3f}", (w // 2 - 230, rt_y + 40), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Pitch: {pitch:.1f} deg", (w // 2 - 230, rt_y + 60), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)
                        cv2.putText(frame, f"FPS: {int(fps)}", (w // 2 - 230, rt_y + 80), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('c'):
                            if audio_manager.play_calibration_start():
                                state = "CALIBRATING"
                                calibration_data = []
                                calibration_start_time = time.time()
                                print("[INFO] Memulai kalibrasi manual...")
                            else:
                                print("[INFO] Tunggu instruksi awal selesai sebelum kalibrasi.")

                    elif state == "CALIBRATING":
                        elapsed_calibration = time.time() - calibration_start_time
                        countdown = max(0, 3 - int(elapsed_calibration))

                        cv2.rectangle(frame, (w // 2 - 180, h // 2 - 100), (w // 2 + 180, h // 2 + 100), (30, 30, 30), -1)
                        cv2.rectangle(frame, (w // 2 - 180, h // 2 - 100), (w // 2 + 180, h // 2 + 100), COLOR_SAFE, 2)
                        cv2.putText(frame, "KALIBRASI BERJALAN", (w // 2 - 70, h // 2 - 70), font, font_normal, COLOR_SAFE, 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Sisa: {countdown} detik", (w // 2 - 50, h // 2 - 30), font, font_normal, COLOR_INFO, 2, cv2.LINE_AA)

                        progress_filled = int((elapsed_calibration / 3.0) * 200)
                        cv2.rectangle(frame, (w // 2 - 100, h // 2 + 10), (w // 2 - 100 + progress_filled, h // 2 + 20), COLOR_SAFE, -1)

                        calibration_data.append({'ear': ear, 'mar': mar, 'pitch': pitch})
                        cv2.putText(frame, f"Sampel: {len(calibration_data)} frame", (w // 2 - 90, h // 2 + 40), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)

                        if elapsed_calibration >= 3.0 and len(calibration_data) > 0:
                            baseline_ear = float(np.mean([d['ear'] for d in calibration_data]))
                            baseline_mar = float(np.mean([d['mar'] for d in calibration_data]))
                            baseline_pitch = float(np.median([d['pitch'] for d in calibration_data]))

                            THRESH_EAR_CLOSE = min(baseline_ear * 0.70, 0.22)
                            THRESH_EAR_BLINK = THRESH_EAR_CLOSE * 1.15
                            THRESH_MAR_YAWN = max(0.20, baseline_mar * 1.3)

                            baseline_values = {'ear': baseline_ear, 'mar': baseline_mar, 'pitch': baseline_pitch}
                            print(f"[INFO] Kalibrasi selesai! Base PITCH: {baseline_pitch:.1f}")

                            detector = DrowsinessDetector(
                                fps=fps,
                                baseline_pitch=baseline_pitch,
                                baseline_ear=baseline_ear,
                                baseline_mouth=baseline_mar,
                                th_ear_close=THRESH_EAR_CLOSE,
                                th_ear_open=THRESH_EAR_BLINK,
                                th_mar_yawn=THRESH_MAR_YAWN
                            )
                            state = "MONITORING"
                            audio_manager.play_calibration_success()
                            calibration_data.clear()

                    elif state == "MONITORING" and detector is not None:
                        result = detector.update(ear, mar, pitch, fps)
                        audio_manager.handle_monitoring_result(result)

                        overlay = frame.copy()
                        hud_top = 35
                        hud_bottom = 22
                        if ui_compact_mode:
                            hud_w_left = int(w * 0.30)
                            hud_w_right = int(w * 0.27)
                            hud_font_scale = 0.38
                            hud_line_h = 12
                        else:
                            hud_w_left = int(w * 0.36)
                            hud_w_right = int(w * 0.34)
                            hud_font_scale = 0.42
                            hud_line_h = 14

                        cv2.rectangle(overlay, (0, hud_top), (hud_w_left, h - hud_bottom), (0, 0, 0), -1)
                        cv2.rectangle(overlay, (w - hud_w_right, hud_top), (w, h - hud_bottom), (0, 0, 0), -1)
                        alpha = 0.68
                        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                        pitch_color = COLOR_INFO
                        if result['head_down_raw']:
                            pitch_color = COLOR_DANGER
                        elif result['head_up']:
                            pitch_color = COLOR_WARN

                        if ui_compact_mode:
                            info_lines = [
                                ("=== QUICK ===", COLOR_INFO),
                                (f"EAR {result['ear']:.3f} | MAR {result['mar']:.3f}", COLOR_SAFE),
                                (f"PITCH {result['pitch']:.1f} | DEV {result['pitch_deviation']:.1f}", pitch_color),
                                (f"PERCLOS {result['perclos']:.1f}%", COLOR_WARN if result['perclos'] > 20 else COLOR_SAFE),
                                (f"B:{result['blink_count']} Y:{result['total_yawns']} H:{result['headdrop_count']} M:{result['microsleep_count']}", COLOR_INFO),
                                (f"MAcc {result['mouth_acc']:.1f} | HAcc {result['head_acc']:.1f}", COLOR_WARN),
                                (f"EAcc {result['eye_acc']:.1f} | MicAcc {result['microsleep_acc']:.1f}", COLOR_DANGER if result['microsleep'] else COLOR_WARN),
                                (f"Score I:{result['instant_score']:.1f} A:{result['accumulator_score']:.1f}", COLOR_DEBUG),
                            ]
                        else:
                            info_lines = [
                                ("=== SENSOR ===", COLOR_INFO),
                                (f"EAR: {result['ear']:.3f}", COLOR_DANGER if result['eye_closed'] else COLOR_SAFE),
                                (f"MAR: {result['mar']:.3f}", COLOR_DANGER if result['yawning'] else COLOR_SAFE),
                                (f"PITCH: {result['pitch']:.1f} deg", pitch_color),
                                (f"Dev: {result['pitch_deviation']:.1f} deg", pitch_color),
                                ("", COLOR_INFO),
                                ("=== THRESH ===", COLOR_INFO),
                                (f"EARTh: {THRESH_EAR_CLOSE:.3f}", COLOR_DEBUG),
                                (f"MARTh: {THRESH_MAR_YAWN:.3f}", COLOR_DEBUG),
                                (f"BaseP: {baseline_values['pitch']:.1f}", COLOR_DEBUG),
                                ("", COLOR_INFO),
                                ("=== YAWN DBG ===", COLOR_INFO),
                                (f"State: {result['yawn_state']}", COLOR_WARN if result['yawn_state'] != 'IDLE' else COLOR_INFO),
                                (f"Hold:{result['yawn_hold_frames']} Pk:{result['yawn_peak_mar']:.2f}", COLOR_INFO),
                                (f"Talk: {'YA' if result['is_talking'] else 'TIDAK'}", COLOR_INFO),
                                ("", COLOR_INFO),
                                ("=== PERCLOS ===", COLOR_INFO),
                                (f"PERCLOS: {result['perclos']:.1f}%", COLOR_WARN if result['perclos'] > 20 else COLOR_SAFE),
                                (f"Win:{result['window_frames']} Cls:{result['closed_frames']}", COLOR_INFO),
                                ("", COLOR_INFO),
                                ("=== COUNTERS ===", COLOR_INFO),
                                (f"Blink: {result['blink_count']}", COLOR_INFO),
                                (f"Yawn: {result['total_yawns']}", COLOR_INFO),
                                (f"HeadDrop: {result['headdrop_count']}", COLOR_INFO),
                                (f"Microsleep: {result['microsleep_count']}", COLOR_INFO),
                                ("", COLOR_INFO),
                                ("=== TEMP ACC ===", COLOR_INFO),
                                (f"M:{result['mouth_acc']:.1f} Y10:{result['recent_yawns_10m']}", COLOR_WARN),
                                (f"H:{result['head_acc']:.1f} H10:{result['recent_head_drops_10m']}", COLOR_WARN),
                                (f"E:{result['eye_acc']:.1f} Mic:{result['microsleep_acc']:.1f}", COLOR_DANGER if result['microsleep'] else COLOR_WARN),
                                (f"Mic10m: {result['recent_microsleeps_10m']}", COLOR_WARN),
                                (f"Inst:{result['instant_score']:.1f} Acc:{result['accumulator_score']:.1f}", COLOR_DEBUG),
                            ]

                        panel_x, panel_y = 8, 50
                        for i, (text, col) in enumerate(info_lines):
                            y_text = panel_y + i * hud_line_h
                            if y_text > (h - hud_bottom - 6):
                                break
                            cv2.putText(frame, text, (panel_x, y_text), font, hud_font_scale, col, 1, cv2.LINE_AA)

                        status_x = w - hud_w_right + 8
                        status_y = 45
                        status_w = hud_w_right - 16

                        if result['microsleep']:
                            st_txt, st_col = "MICROSLEEP", COLOR_DANGER
                        elif result['score'] >= LEVEL_BAHAYA:
                            st_txt, st_col = "BAHAYA", COLOR_DANGER
                        elif result['score'] >= LEVEL_WASPADA:
                            st_txt, st_col = "WASPADA", COLOR_WARN
                        elif result['score'] >= LEVEL_HATI_HATI:
                            st_txt, st_col = "HATI-HATI", (0, 255, 255)
                        else:
                            st_txt, st_col = "AMAN", COLOR_SAFE

                        cv2.rectangle(frame, (status_x, status_y), (status_x + status_w, status_y + 54), st_col, 1)
                        cv2.putText(frame, "SYSTEM STATUS", (status_x + 4, status_y + 13), font, 0.40, st_col, 1, cv2.LINE_AA)
                        cv2.putText(frame, st_txt, (status_x + 4, status_y + 33), font, 0.55, st_col, 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Score: {int(result['score'])}/100", (status_x + 4, status_y + 49), font, 0.40, st_col, 1, cv2.LINE_AA)

                        roi_y = status_y + 64
                        roi_size = min(100, status_w - 10) if not ui_compact_mode else min(76, status_w - 10)
                        if roi_frame is not None:
                            roi_resized = cv2.resize(roi_frame, (roi_size, roi_size))
                            roi_x = status_x + (status_w - roi_size) // 2
                            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), COLOR_ROI, 1)
                            frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size] = roi_resized
                            cv2.putText(frame, "ROI", (roi_x, roi_y - 6), font, 0.40, COLOR_ROI, 1, cv2.LINE_AA)

                        graph_y_start = roi_y + roi_size + 18
                        color_mar_graph = (0, 255, 255)
                        color_pitch_graph = (0, 165, 255)
                        graph_h = 42 if not ui_compact_mode else 34
                        graph_gap = 12 if not ui_compact_mode else 8

                        if show_graphs:
                            for idx, (h_list, th, name, col) in enumerate([
                                (ear_history, THRESH_EAR_CLOSE, "EAR History", COLOR_SAFE),
                                (mar_history, THRESH_MAR_YAWN, "MAR History", color_mar_graph),
                                (pitch_history, baseline_values['pitch'], "Pitch History", color_pitch_graph)
                            ]):
                                g_y = graph_y_start + idx * (graph_h + graph_gap)
                                if g_y + graph_h > (h - hud_bottom - 2):
                                    break
                                cv2.rectangle(frame, (status_x, g_y), (status_x + status_w, g_y + graph_h), col, 1)
                                cv2.putText(frame, name, (status_x + 3, g_y - 4), font, 0.38, col, 1, cv2.LINE_AA)

                                if len(h_list) > 1:
                                    mx = max(h_list) if h_list else 1
                                    mn = min(h_list) if h_list else 0
                                    rg = mx - mn if mx > mn else 1.0
                                    gw = status_w - 2
                                    pts = [
                                        (
                                            status_x + 1 + int((i / max(1, len(h_list) - 1)) * (gw - 2)),
                                            int(g_y + graph_h - ((v - mn) / rg) * graph_h)
                                        )
                                        for i, v in enumerate(h_list)
                                    ]
                                    pts = [(x, max(g_y + 1, min(y, g_y + graph_h - 1))) for x, y in pts]
                                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)
                                    th_y = int(g_y + graph_h - ((th - mn) / rg) * graph_h)
                                    if g_y <= th_y <= g_y + graph_h:
                                        cv2.line(frame, (status_x, th_y), (status_x + status_w, th_y), (120, 120, 120), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, "Graph: OFF (press H)", (status_x + 4, graph_y_start + 14), font, 0.40, COLOR_INFO, 1, cv2.LINE_AA)

                        if result['microsleep']:
                            if int(time.time() * 4) % 2 == 0:
                                over = frame.copy()
                                cv2.rectangle(over, (0, 0), (w, h), (0, 0, 255), -1)
                                cv2.addWeighted(over, 0.15, frame, 0.85, 0, frame)
                            draw_text_with_outline(frame, "!!! MICROSLEEP DETECTED !!!", (w // 2 - 180, h // 2), font, 0.7, COLOR_DANGER, 2)
                        elif result['score'] >= LEVEL_BAHAYA:
                            if int(time.time() * 2) % 2 == 0:
                                over = frame.copy()
                                cv2.rectangle(over, (0, 0), (w, h), (0, 0, 255), -1)
                                cv2.addWeighted(over, 0.1, frame, 0.9, 0, frame)
                            draw_text_with_outline(frame, "!!! PERINGATAN KELELAHAN !!!", (w // 2 - 180, h // 2), font, 0.7, COLOR_DANGER, 2)
                        elif result['score'] >= LEVEL_WASPADA:
                            draw_text_with_outline(frame, "WASPADA - Tanda Kelelahan", (w // 2 - 150, h // 2 - 50), font, 0.6, COLOR_WARN, 2)

                        if result['yawning']:
                            draw_text_with_outline(frame, "MENGUAP TERDETEKSI", (w // 2 - 120, h // 2 + 30), font, 0.6, (0, 200, 255), 2)
                        if result['head_down_raw']:
                            draw_text_with_outline(frame, "KEPALA MENUNDUK", (w // 2 - 100, h // 2 + 60), font, 0.6, COLOR_DANGER, 2)

            if not has_face:
                cv2.rectangle(frame, (w // 2 - 230, h // 2 - 30), (w // 2 + 230, h // 2 + 30), (30, 30, 30), -1)
                cv2.rectangle(frame, (w // 2 - 230, h // 2 - 30), (w // 2 + 230, h // 2 + 30), COLOR_WARN, 1)
                draw_text_with_outline(frame, "MENCARI WAJAH...", (w // 2 - 130, h // 2 + 10), font, 0.7, COLOR_WARN, 2)

            cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
            cv2.putText(frame, f"Raspberry Pi 5 DROWSINESS | FPS: {int(fps)} | State: {state}", (10, 22), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, h - 22), (w, h), (20, 20, 20), -1)
            cv2.putText(frame, "q:Keluar | c:Kalibrasi | r:Reset | s:Screenshot | h:Graph | u:UI", (10, h - 5), font, font_small, COLOR_INFO, 1, cv2.LINE_AA)

            cv2.imshow('Deteksi Kelelahan RPi5 (AUDIO MANAGER - rpicam-vid)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                show_graphs = not show_graphs
            elif key == ord('u'):
                ui_compact_mode = not ui_compact_mode
            elif key == ord('r'):
                state = "CALIBRATION"
                detector = None
                calibration_data.clear()
                ear_history.clear()
                mar_history.clear()
                pitch_history.clear()
                audio_manager.reset_for_calibration()
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_rpi_{time.strftime('%Y%m%d_%H%M%S')}.png", frame)
    except BaseException as e:
        traceback.print_exc()
        print(f"[ERROR] Process dihentikan: {e}")
    finally:
        print("[INFO] Membersihkan resource...")
        audio_manager.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
