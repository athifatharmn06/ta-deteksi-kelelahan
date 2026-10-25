import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import threading
from queue import Queue, Full, Empty
import re
import sys
import math
import subprocess
from collections import deque

# ============================================
# KONFIGURASI PENGUJIAN (LOKASI RPI 5)
# ============================================
HOME_DIR = os.path.expanduser("~")
BASE_DIR = os.path.join(HOME_DIR, "TA Deteksi Kelelahan", "Laporan Progress", "Pengujian 1 dan 2")
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(BASE_DIR, "video")
PERF_DIR = os.path.join(BASE_DIR, "performa")
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(PERF_DIR, exist_ok=True)

def get_next_subject_id(video_dir, perf_dir):
    pattern = re.compile(r"(?:video|performa) subjek (\d+) - pengujian 1 dan 2", re.IGNORECASE)
    max_id = 0
    for folder in (video_dir, perf_dir):
        try:
            for name in os.listdir(folder):
                match = pattern.search(name)
                if match:
                    max_id = max(max_id, int(match.group(1)))
        except FileNotFoundError:
            continue
    return max_id + 1

def create_video_writer(path, size, fps):
    fourcc_candidates = ["MJPG", "XVID", "mp4v"]
    for code in fourcc_candidates:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, size)
        if writer.isOpened():
            return writer, code
        writer.release()
    return None, None

class VideoRecorder:
    def __init__(self, path, size, fps, queue_size=120):
        self.path = path
        self.size = size
        self.fps = fps
        self.queue = Queue(maxsize=queue_size)
        self.writer = None
        self.codec = None
        self.thread = None
        self.running = False

    def start(self):
        self.writer, self.codec = create_video_writer(self.path, self.size, self.fps)
        if self.writer is None:
            return False
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.writer:
            self.writer.release()
        self.writer = None
        self.codec = None

    def enqueue(self, frame):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame)
        except Full:
            # Drop frame when overloaded to keep UI responsive
            pass

    def _worker(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except Empty:
                continue
            if self.writer is not None:
                self.writer.write(frame)

# ============================================
# KONFIGURASI SISTEM (Ground Truth: v5_own_detection_model.py)
# ============================================
W_EYE = 0.45        
W_MOUTH = 0.25      
W_HEAD = 0.30       

THRESH_EAR_CLOSE = 0.18      
THRESH_MAR_YAWN = 0.20       
THRESH_HEAD_PITCH = 10.0     # using 10° head-pitch threshold per literature

LEVEL_HATI_HATI = 20
LEVEL_WASPADA   = 40
LEVEL_BAHAYA    = 60

TIME_MICROSLEEP_LIMIT = 1.5  
TIME_YAWN_MIN_DURATION = 1.5  
TIME_BLINK_MIN_INTERVAL = 0.2  
TIME_CONSECUTIVE_FRAMES = 15  
TIME_PERCLOS_WINDOW = 60      

USE_OSCILLATION_GUARD = True   
USE_PEAK_GATE         = True   
USE_PERCLOS_WARMUP    = True   

# UI AESTHETICS (V4 STYLE)
COLOR_SAFE = (0, 255, 0)     
COLOR_WARN = (0, 165, 255)   
COLOR_DANGER = (0, 0, 255)   
COLOR_INFO = (200, 200, 200) 
COLOR_DEBUG = (255, 255, 0)  
COLOR_ROI = (0, 255, 255)    
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================
# KAMERA RASPBERRY PI 5
# ============================================
class RPiCameraFeed:
    def __init__(self, tuning_file="/usr/share/libcamera/ipa/rpi/pisp/imx219_noir.json", width=640, height=480, fps=30, capture_width=1280, capture_height=720):
        self.width, self.height = width, height
        self.capture_width, self.capture_height = capture_width, capture_height
        self.use_rpicam = True
        self.process = None
        self.cap = None
        try:
            cmd = ["rpicam-vid", "-t", "0", "--tuning-file", tuning_file, "--width", str(capture_width), "--height", str(capture_height), "--framerate", str(fps), "--codec", "yuv420", "--inline", "--nopreview", "-o", "-"]
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            self.frame_size = int(capture_width * capture_height * 1.5)
            time.sleep(0.5)
            if self.process.poll() is not None: raise Exception()
        except:
            self.use_rpicam = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, capture_width); self.cap.set(4, capture_height)

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
        x = (self.width - new_w) // 2
        y = (self.height - new_h) // 2
        canvas[y:y+new_h, x:x+new_w] = resized
        return canvas

    def isOpened(self): return (self.process is not None and self.process.poll() is None) if self.use_rpicam else (self.cap and self.cap.isOpened())
    def read(self):
        if self.use_rpicam:
            raw = self.process.stdout.read(self.frame_size)
            if not raw or len(raw) != self.frame_size: return False, None
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.capture_height * 3 // 2, self.capture_width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, self._letterbox(frame)
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, self._letterbox(frame)
    def release(self):
        if self.use_rpicam: self.process.terminate()
        else: self.cap.release()

# ============================================
# DROWSINESS ACCUMULATOR (Ground Truth)
# ============================================
class DrowsinessAccumulator:
    def __init__(self):
        self.mouth_acc = self.head_acc = self.eye_acc = self.microsleep_acc = 0.0
        # Reduce decay to retain accumulator contribution longer after events
        self.mouth_decay_per_sec = 0.08
        self.head_decay_per_sec = 0.10
        self.eye_decay_per_sec = 0.06
        self.microsleep_decay_per_sec = 0.025
        # Increase event gains
        self.mouth_event_gain = 30.0
        self.head_event_gain = 25.0
        self.microsleep_event_gain = 60.0
        self.head_hold_gain_per_sec = 3.0
        self.eye_hold_gain_per_sec = 2.6
        self.perclos_gain_per_sec = 0.9
        self.microsleep_hold_gain_per_sec = 8.0
        self.recent_yawns = deque(); self.recent_head_drops = deque(); self.recent_microsleeps = deque()
        self.recent_window_sec = 600.0
        self.microsleep_count = 0

    def update(self, dt, now_ts, h_down, e_closed, perclos, micro):
        scale = 0.55 if (h_down or e_closed or perclos > 20.0 or micro) else 1.0
        self.mouth_acc = max(0, self.mouth_acc - self.mouth_decay_per_sec * dt)
        self.head_acc = max(0, self.head_acc - self.head_decay_per_sec * dt * scale)
        self.eye_acc = max(0, self.eye_acc - self.eye_decay_per_sec * dt * scale)
        self.microsleep_acc = max(0, self.microsleep_acc - self.microsleep_decay_per_sec * dt * scale)
        if h_down: self.head_acc = min(100, self.head_acc + self.head_hold_gain_per_sec * dt)
        if e_closed: self.eye_acc = min(100, self.eye_acc + self.eye_hold_gain_per_sec * dt)
        if perclos > 20.0: self.eye_acc = min(100, self.eye_acc + self.perclos_gain_per_sec * dt * (perclos/20.0))
        if micro: self.microsleep_acc = min(100, self.microsleep_acc + self.microsleep_hold_gain_per_sec * dt)
        for d in [self.recent_yawns, self.recent_head_drops, self.recent_microsleeps]:
            while d and (now_ts - d[0]) > self.recent_window_sec: d.popleft()

    def add_yawn(self, ts): self.mouth_acc = min(100, self.mouth_acc + self.mouth_event_gain); self.recent_yawns.append(ts)
    def add_head_drop(self, ts): self.head_acc = min(100, self.head_acc + self.head_event_gain); self.recent_head_drops.append(ts)
    def add_microsleep(self, ts):
        self.microsleep_acc = min(100, self.microsleep_acc + self.microsleep_event_gain)
        self.recent_microsleeps.append(ts)
        self.microsleep_count += 1
    def get_eye_score(self): return min(100, self.eye_acc + min(45, self.microsleep_acc*0.55 + len(self.recent_microsleeps)*10))
    def get_mouth_score(self): return min(100, self.mouth_acc + min(40, len(self.recent_yawns)*8))
    def get_head_score(self): return min(100, self.head_acc + min(40, len(self.recent_head_drops)*10))

# ============================================
# DROWSINESS DETECTOR (V5 LOGIC)
# ============================================
class DrowsinessDetector:
    def __init__(self, fps=30, baseline_pitch=0.0):
        self.eb, self.mb, self.pb = [deque(maxlen=5) for _ in range(3)]
        self.ecf = self.hdf = self.vbc = 0
        self.bip = False; self.bfc = 0
        self.ys = 'IDLE'; self.yhf = 0; self.ypm = 0.0; self.lyt = 0.0
        self.mct = deque(); self.map = False
        self.perclos_calc = PERCLOSCalculator(60, fps); self.acc = DrowsinessAccumulator()
        self.bp = baseline_pitch; self.lut = time.time(); self.ms = None; self.pm = self.ph = False

    def update(self, ear, mar, pitch, fps):
        def sm(b, v): b.append(v); return sum(b)/len(b)
        es, ms, ps = sm(self.eb, ear), sm(self.mb, mar), sm(self.pb, pitch)
        p_data = self.perclos_calc.update(es, THRESH_EAR_CLOSE)
        ec = es < THRESH_EAR_CLOSE
        if ec and not self.bip: self.bip, self.bfc = True, 1
        elif ec: self.bfc += 1
        elif self.bip:
            if 2 <= self.bfc <= 15: self.vbc += 1
            self.bip, self.bfc = False, 0
        pd = ps - self.bp
        hp_th = THRESH_HEAD_PITCH
        hdr = pd < -hp_th and ps < -3.0
        hu = pd > hp_th and ps > 3.0
        m_ab = ms > THRESH_MAR_YAWN; t = time.time()
        if m_ab and not self.map: self.mct.append(t)
        while self.mct and (t - self.mct[0]) > 4.0: self.mct.popleft()
        is_t = len(self.mct) > 3; self.map = m_ab
        if self.ys == 'IDLE' and m_ab and not is_t: self.ys, self.yhf, self.ypm = 'BUILDING', 1, ms
        elif self.ys == 'BUILDING':
            if not m_ab or is_t: self.ys, self.yhf = 'IDLE', 0
            else:
                self.yhf += 1; self.ypm = max(self.ypm, ms)
                if self.yhf >= int(1.5*fps) and self.ypm >= max(THRESH_MAR_YAWN*1.6, 0.5):
                    self.ys, self.lyt = 'CONFIRMED', t; self.acc.add_yawn(t)
        elif self.ys == 'CONFIRMED' and not m_ab: self.ys = 'COOLDOWN'
        elif self.ys == 'COOLDOWN' and (t - self.lyt) >= 5.0: self.ys = 'IDLE'
        if ec and self.ys != 'CONFIRMED' and ms < THRESH_MAR_YAWN*0.7: self.ecf = min(self.ecf+1, 60)
        else: self.ecf = max(self.ecf-2, 0)
        m_det = False
        if self.ecf >= 15:
            if self.ms is None: self.ms = t
            elif (t - self.ms) > 1.5: m_det = True
        else: self.ms = None
        if hdr: self.hdf = min(self.hdf+1, 30)
        else: self.hdf = max(self.hdf-1, 0)
        hd_det = self.hdf >= 15
        dt = min(t - self.lut, 0.5); self.lut = t
        if m_det and not self.pm: self.acc.add_microsleep(t)
        if hd_det and not self.ph: self.acc.add_head_drop(t)
        self.pm, self.ph = m_det, hd_det
        self.acc.update(dt, t, hdr, ec, p_data['perclos'], m_det)
        ins = 0
        if m_det: ins += W_EYE*100
        elif self.ecf >= 15: ins += W_EYE*60
        elif p_data['perclos'] > 20: ins += W_EYE * (min(p_data['perclos'],40)/40*70)
        if self.ys == 'CONFIRMED': ins += W_MOUTH*70
        if hd_det: ins += W_HEAD*100
        elif hu: ins += W_HEAD*30
        ea, ma, ha = self.acc.get_eye_score(), self.acc.get_mouth_score(), self.acc.get_head_score()
        sev = 0.25*max(ea,ma,ha); tev = len(self.acc.recent_microsleeps) + len(self.acc.recent_yawns) + len(self.acc.recent_head_drops)
        if tev > 1: sev += (tev-1)*10.0
        ascore = min(100, W_EYE*ea + W_MOUTH*ma + W_HEAD*ha + sev)
        fs = 0.35*ins + 0.65*ascore
        return {'ear':es,'mar':ms,'pitch':ps,'pitch_deviation':pd,'perclos':p_data['perclos'],'eye_closed':ec,'microsleep':m_det,'yawning':self.ys=='CONFIRMED','head_down':hd_det,'head_down_raw':hdr,'head_up':hu,'score':fs,'vbc':self.vbc,'yawn_count':len(self.acc.recent_yawns),'micro_count':len(self.acc.recent_microsleeps),'microsleep_count':self.acc.microsleep_count,'yawn_state':self.ys,'is_talking':is_t,'mouth_acc':self.acc.mouth_acc,'head_acc':self.acc.head_acc,'eye_acc':self.acc.eye_acc,'microsleep_acc':self.acc.microsleep_acc,'inst':ins,'acc':ascore}

class PERCLOSCalculator:
    def __init__(self, ws=60, fps=30):
        self.ws = int(ws*fps); self.es = deque(maxlen=self.ws); self.bc = 0; self.lbt = time.time(); self.br = 0; self.les = False
    def update(self, ear, th):
        c = ear < th
        if self.les and not c: self.bc += 1
        self.les = c; self.es.append(c); p = (sum(self.es)/len(self.es)*100) if len(self.es)>=self.ws//2 else 0
        t = time.time()
        if (t-self.lbt)>=60: self.br = self.bc; self.bc=0; self.lbt=t
        return {'perclos':p,'blink_count':self.bc}

# ============================================
# UI UTILS (V4 STYLE)
# ============================================
def draw_text_outlined(img, text, pos, scale, col, thick=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, FONT, scale, col, thick, cv2.LINE_AA)

def main():
    cap = RPiCameraFeed(); w, h = cap.width, cap.height
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.8)
    state = "READY"; subject_id = get_next_subject_id(VIDEO_DIR, PERF_DIR)
    video_path = os.path.join(VIDEO_DIR, f"video subjek {subject_id} - pengujian 1 dan 2.avi")
    perf_path = os.path.join(PERF_DIR, f"performa subjek {subject_id} - pengujian 1 dan 2.csv")
    while os.path.exists(video_path) or os.path.exists(perf_path):
        subject_id += 1
        video_path = os.path.join(VIDEO_DIR, f"video subjek {subject_id} - pengujian 1 dan 2.avi")
        perf_path = os.path.join(PERF_DIR, f"performa subjek {subject_id} - pengujian 1 dan 2.csv")
    
    det, bv = None, {}; fps_ema, pt = 30.0, time.time()
    c_start, c_data, flip_vertical = None, [], False; show_g, compact = True, False
    baseline_values = {'pitch': 0.0}
    eh, mh, ph = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    recorder = None
    perf_file, perf_writer = None, None
    recording, rec_start, last_perf_log = False, None, 0.0
    rec_codec = None

    print(f"[INFO] Pengujian V5 Premium HUD. Subjek: {subject_id}")
    while cap.isOpened():
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret: break
        if flip_vertical:
            frame = cv2.flip(frame, 0)
        raw = frame.copy()
        t = time.time(); dt = t-pt; fps_ema = 0.9*fps_ema + 0.1*(1/(dt+1e-6)); pt = t
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            def get_p(i): return (int(lm[i].x*w), int(lm[i].y*h))
            def dist(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            def calc_ear():
                def eye(idx):
                    p = [get_p(i) for i in idx]
                    return (dist(p[1],p[5])+dist(p[2],p[4]))/(2*dist(p[0],p[3]))
                return (eye([33,160,158,133,153,144])+eye([362,385,387,263,373,380]))/2
            def calc_mar():
                p = [get_p(i) for i in [61,291,82,87,13,14,312,317]]
                return (dist(p[2],p[3])+dist(p[4],p[5])+dist(p[6],p[7]))/(2*dist(p[0],p[1]))
            def calc_pose():
                p10,p152,p4 = get_p(10),get_p(152),get_p(4)
                ax,ay = p152[0]-p10[0], p152[1]-p10[1]; nx,ny = p4[0]-p10[0], p4[1]-p10[1]
                f = ax*ax+ay*ay
                return max(-45,min(45,(0.45-(nx*ax+ny*ay)/f)*120)) if f>1 else 0
            
            ear, mar, pose = calc_ear(), calc_mar(), calc_pose()
            eh.append(ear); mh.append(mar); ph.append(pose)
            
            xc, yc = [l.x for l in lm], [l.y for l in lm]
            x1, y1, x2, y2 = int(min(xc)*w)-15, int(min(yc)*h)-15, int(max(xc)*w)+15, int(max(yc)*h)+15
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_ROI, 1)
            roi = raw[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

            if state == "READY":
                cv2.rectangle(frame, (w//2-250, h//2-130), (w//2+250, h//2+200), (30, 30, 30), -1)
                cv2.rectangle(frame, (w//2-250, h//2-130), (w//2+250, h//2+200), COLOR_WARN, 2)
                cv2.putText(frame, "KALIBRASI WAJAH", (w//2-80, h//2-100), FONT, 0.65, COLOR_WARN, 2, cv2.LINE_AA)
                instructions = [
                    "1. Hadap kamera dengan posisi normal",
                    "2. Mata terbuka normal",
                    "3. Mulut tertutup",
                    "4. Kepala lurus ke depan",
                    "5. Tekan 'C' untuk mulai (3 detik)",
                    "6. Tunggu hingga selesai"
                ]
                for i, text in enumerate(instructions):
                    cv2.putText(frame, text, (w//2-230, h//2 - 60 + i*22), FONT, 0.55, COLOR_INFO, 1, cv2.LINE_AA)
                rt_y = h//2 + 90
                cv2.putText(frame, "=== NILAI REAL-TIME ===", (w//2-230, rt_y), FONT, 0.50, COLOR_DEBUG, 1, cv2.LINE_AA)
                cv2.putText(frame, f"EAR: {ear:.3f}", (w//2-230, rt_y+20), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                cv2.putText(frame, f"MAR: {mar:.3f}", (w//2-230, rt_y+40), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                cv2.putText(frame, f"Pitch: {pose:.1f} deg", (w//2-230, rt_y+60), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {int(fps_ema)}", (w//2-230, rt_y+80), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    state = "CALIBRATING"
                    c_start = time.time()
                    c_data = []
                elif key == ord('f'):
                    flip_vertical = not flip_vertical
            elif state == "CALIBRATING":
                elapsed_calibration = time.time() - c_start
                countdown = max(0, 3 - int(elapsed_calibration))
                cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), (30, 30, 30), -1)
                cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), COLOR_SAFE, 2)
                cv2.putText(frame, "KALIBRASI BERJALAN", (w//2-70, h//2-70), FONT, 0.65, COLOR_SAFE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Sisa: {countdown} detik", (w//2-50, h//2-30), FONT, 0.65, COLOR_INFO, 2, cv2.LINE_AA)
                progress_filled = int((elapsed_calibration / 3.0) * 200)
                cv2.rectangle(frame, (w//2-100, h//2+10), (w//2-100 + progress_filled, h//2+20), COLOR_SAFE, -1)
                c_data.append({'e':ear, 'm':mar, 'p':pose})
                cv2.putText(frame, f"Sampel: {len(c_data)} frame", (w//2-90, h//2+40), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                if elapsed_calibration >= 3.0 and len(c_data) > 0:
                    be = np.mean([x['e'] for x in c_data])
                    bm = np.mean([x['m'] for x in c_data])
                    bp = np.median([x['p'] for x in c_data])
                    global THRESH_EAR_CLOSE, THRESH_MAR_YAWN
                    THRESH_EAR_CLOSE = min(be * 0.72, 0.22)
                    THRESH_MAR_YAWN = max(0.2, bm * 1.3)
                    det = DrowsinessDetector(fps=int(fps_ema), baseline_pitch=bp)
                    bv = {'p': bp}
                    baseline_values['pitch'] = bp
                    state = "MONITORING"
            elif state=="MONITORING" and det:
                r = det.update(ear, mar, pose, int(fps_ema))
                # normalize keys used by v4 UI
                r['total_yawns'] = r.get('yawn_count', 0)
                r['valid_blink_count'] = r.get('vbc', 0)
                r['blink_count'] = r.get('vbc', 0)
                r['yawn_peak_mar'] = r.get('ypm', r.get('yawn_peak_mar', 0.0))
                r['yawn_hold_frames'] = r.get('yhf', r.get('yawn_hold_frames', 0))
                if recording and recorder is not None:
                    recorder.enqueue(raw)
                    latency_ms = (time.time() - frame_start) * 1000.0
                    if perf_writer and (t - last_perf_log) >= 1.0:
                        perf_writer.writerow([time.strftime("%H:%M:%S"), round(t - rec_start, 1), int(fps_ema), round(latency_ms, 2)])
                        perf_file.flush()
                        last_perf_log = t

                # HUD/UI adapted from v4 (UI/UX only)
                ui_compact_mode = compact
                show_graphs = show_g

                ov = frame.copy()
                if ui_compact_mode:
                    hl, hr = int(w * 0.30), int(w * 0.27)
                    hud_font_scale = 0.38
                    hud_line_h = 12
                else:
                    hl, hr = int(w * 0.36), int(w * 0.34)
                    hud_font_scale = 0.42
                    hud_line_h = 14

                cv2.rectangle(ov, (0, 35), (hl, h-22), (0,0,0), -1)
                cv2.rectangle(ov, (w-hr, 35), (w, h-22), (0,0,0), -1)
                cv2.addWeighted(ov, 0.68, frame, 0.32, 0, frame)

                pitch_color = COLOR_INFO
                if r['head_down_raw']: pitch_color = COLOR_DANGER
                elif r['head_up']: pitch_color = COLOR_WARN
                hp_th_ui = THRESH_HEAD_PITCH
                hp_low = baseline_values.get('pitch', 0.0) - hp_th_ui

                if ui_compact_mode:
                    info_lines = [
                        ("=== QUICK ===", COLOR_INFO),
                        (f"EAR {r['ear']:.3f} | MAR {r['mar']:.3f}", COLOR_SAFE),
                        (f"PITCH {r['pitch']:.1f} | DEV {r['pitch_deviation']:.1f}", pitch_color),
                        (f"PERCLOS {r['perclos']:.1f}%", COLOR_WARN if r['perclos']>20 else COLOR_SAFE),
                        (f"Blink {r.get('blink_count',0)} | Yawn {r.get('total_yawns',0)}", COLOR_INFO),
                        (f"MAcc {r['mouth_acc']:.1f} | HAcc {r['head_acc']:.1f}", COLOR_WARN),
                        (f"EAcc {r['eye_acc']:.1f} | MicAcc {r['microsleep_acc']:.1f}", COLOR_DANGER if r['microsleep'] else COLOR_WARN),
                        (f"Score I:{r['inst']:.1f} A:{r['acc']:.1f}", COLOR_DEBUG),
                    ]
                else:
                    info_lines = [
                        ("=== SENSOR ===", COLOR_INFO),
                        (f"EAR: {r['ear']:.3f}", COLOR_DANGER if r['eye_closed'] else COLOR_SAFE),
                        (f"MAR: {r['mar']:.3f}", COLOR_DANGER if r['yawning'] else COLOR_SAFE),
                        (f"PITCH: {r['pitch']:.1f} deg", pitch_color),
                        (f"Dev: {r['pitch_deviation']:.1f} deg", pitch_color),
                        ("", COLOR_INFO),
                        ("=== THRESH ===", COLOR_INFO),
                        (f"EARTh: {THRESH_EAR_CLOSE:.3f}", COLOR_DEBUG),
                        (f"MARTh: {THRESH_MAR_YAWN:.3f}", COLOR_DEBUG),
                        (f"BaseP: {baseline_values.get('pitch',0.0):.1f}", COLOR_DEBUG),
                        (f"HeadDnTh: {hp_low:.1f}", COLOR_DEBUG),
                        ("", COLOR_INFO),
                        ("=== YAWN DBG ===", COLOR_INFO),
                        (f"State: {r.get('yawn_state','IDLE')}", COLOR_WARN if r.get('yawn_state','IDLE') != 'IDLE' else COLOR_INFO),
                        (f"Hold:{r.get('yawn_hold_frames',0)} Pk:{r.get('yawn_peak_mar',0.0):.2f}", COLOR_INFO),
                        (f"Talk: {'YA' if r.get('is_talking',False) else 'TIDAK'}", COLOR_INFO),
                        ("", COLOR_INFO),
                        ("=== PERCLOS ===", COLOR_INFO),
                        (f"PERCLOS: {r['perclos']:.1f}%", COLOR_WARN if r['perclos']>20 else COLOR_SAFE),
                        (f"Win:{r.get('window_frames',0)} Cls:{r.get('closed_frames',0)}", COLOR_INFO),
                        ("", COLOR_INFO),
                        ("=== COUNTERS ===", COLOR_INFO),
                        (f"BlinkP:{r.get('blink_count',0)} Val:{r.get('valid_blink_count',0)}", COLOR_INFO),
                        (f"Yawns: {r.get('total_yawns',0)}", COLOR_INFO),
                        ("", COLOR_INFO),
                        ("=== TEMP ACC ===", COLOR_INFO),
                        (f"M:{r['mouth_acc']:.1f} Y10:{len(det.acc.recent_yawns)}", COLOR_WARN),
                        (f"H:{r['head_acc']:.1f} H10:{len(det.acc.recent_head_drops)}", COLOR_WARN),
                        (f"E:{r['eye_acc']:.1f} Mic:{r['microsleep_acc']:.1f}", COLOR_DANGER if r['microsleep'] else COLOR_WARN),
                        (f"Mic10m: {len(det.acc.recent_microsleeps)}", COLOR_WARN),
                        (f"Inst:{r['inst']:.1f} Acc:{r['acc']:.1f}", COLOR_DEBUG),
                    ]

                panel_x, panel_y = 8, 50
                for i, (text, col) in enumerate(info_lines):
                    y_text = panel_y + i * hud_line_h
                    if y_text > (h - 22 - 6):
                        break
                    cv2.putText(frame, text, (panel_x, y_text), FONT, hud_font_scale, col, 1, cv2.LINE_AA)

                sx = w - hr + 8
                status_x = sx
                status_y = 45
                status_w = hr - 16

                if r['microsleep']:
                    st_txt, st_col = "MICROSLEEP", COLOR_DANGER
                elif r['score'] >= LEVEL_BAHAYA:
                    st_txt, st_col = "BAHAYA", COLOR_DANGER
                elif r['score'] >= LEVEL_WASPADA:
                    st_txt, st_col = "WASPADA", COLOR_WARN
                elif r['score'] >= LEVEL_HATI_HATI:
                    st_txt, st_col = "HATI-HATI", (0,255,255)
                else:
                    st_txt, st_col = "AMAN", COLOR_SAFE

                cv2.rectangle(frame, (status_x, status_y), (status_x + status_w, status_y + 54), st_col, 1)
                cv2.putText(frame, "SYSTEM STATUS", (status_x+4, status_y+13), FONT, 0.40, st_col, 1, cv2.LINE_AA)
                cv2.putText(frame, st_txt, (status_x+4, status_y+33), FONT, 0.55, st_col, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Score: {int(r['score'])}/100", (status_x+4, status_y+49), FONT, 0.40, st_col, 1, cv2.LINE_AA)
                if recording:
                    rec_elapsed = int(t - rec_start) if rec_start else 0
                    cv2.circle(frame, (status_x + status_w - 12, status_y + 12), 5, COLOR_DANGER, -1)
                    label = f"REC {rec_elapsed}s"
                    if rec_codec:
                        label = f"REC {rec_elapsed}s ({rec_codec})"
                    cv2.putText(frame, label, (status_x+4, status_y - 6), FONT, 0.40, COLOR_DANGER, 1, cv2.LINE_AA)
                elif recorder is None and rec_start is not None:
                    cv2.putText(frame, "REC FAILED", (status_x+4, status_y - 6), FONT, 0.40, COLOR_DANGER, 1, cv2.LINE_AA)

                roi_y = status_y + 64
                roi_size = min(100, status_w - 10) if not ui_compact_mode else min(76, status_w - 10)
                if roi is not None and roi.size:
                    roi_resized = cv2.resize(roi, (roi_size, roi_size))
                    roi_x = status_x + (status_w - roi_size) // 2
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), COLOR_ROI, 1)
                    frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size] = roi_resized
                    cv2.putText(frame, "ROI", (roi_x, roi_y - 6), FONT, 0.40, COLOR_ROI, 1, cv2.LINE_AA)

                graph_y_start = roi_y + roi_size + 18
                COLOR_MAR_GRAPH = (0, 255, 255)
                COLOR_PITCH_GRAPH = (0, 165, 255)
                graph_h = 42 if not ui_compact_mode else 34
                graph_gap = 12 if not ui_compact_mode else 8

                if show_graphs:
                    for idx, (h_list, th, name, col) in enumerate([
                        (eh, THRESH_EAR_CLOSE, "EAR History", COLOR_SAFE),
                        (mh, THRESH_MAR_YAWN, "MAR History", COLOR_MAR_GRAPH),
                        (ph, bv.get('p', 0.0) if isinstance(bv, dict) else 0.0, "Pitch History", COLOR_PITCH_GRAPH)
                    ]):
                        g_y = graph_y_start + idx * (graph_h + graph_gap)
                        if g_y + graph_h > (h - 22 - 2):
                            break
                        cv2.rectangle(frame, (status_x, g_y), (status_x + status_w, g_y + graph_h), col, 1)
                        cv2.putText(frame, name, (status_x+3, g_y-4), FONT, 0.38, col, 1, cv2.LINE_AA)

                        if len(h_list) > 1:
                            mx, mn = max(h_list), min(h_list)
                            rg = mx - mn if mx > mn else 1.0
                            gw = status_w - 2
                            pts = [(status_x + 1 + int((i / max(1, len(h_list)-1)) * (gw - 2)), int(g_y + graph_h - ((v - mn)/rg) * graph_h)) for i, v in enumerate(h_list)]
                            pts = [(x, max(g_y + 1, min(y, g_y + graph_h - 1))) for x, y in pts]
                            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 1, cv2.LINE_AA)
                            th_y = int(g_y + graph_h - ((th - mn)/rg) * graph_h) if rg != 0 else g_y
                            if g_y <= th_y <= g_y + graph_h:
                                cv2.line(frame, (status_x, th_y), (status_x + status_w, th_y), (120, 120, 120), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Graph: OFF (press H)", (status_x + 4, graph_y_start + 14), FONT, 0.40, COLOR_INFO, 1, cv2.LINE_AA)

                if r['microsleep']:
                    if int(time.time() * 4) % 2 == 0:
                        over = frame.copy(); cv2.rectangle(over, (0,0), (w,h), (0,0,255), -1); cv2.addWeighted(over, 0.15, frame, 0.85, 0, frame)
                    draw_text_outlined(frame, "!!! MICROSLEEP DETECTED !!!", (w//2-180, h//2), 0.7, COLOR_DANGER, 2)
                elif r['score'] >= LEVEL_BAHAYA:
                    if int(time.time() * 2) % 2 == 0:
                        over = frame.copy(); cv2.rectangle(over, (0,0), (w,h), (0,0,255), -1); cv2.addWeighted(over, 0.1, frame, 0.9, 0, frame)
                    draw_text_outlined(frame, "!!! PERINGATAN KELELAHAN !!!", (w//2-180, h//2), 0.7, COLOR_DANGER, 2)
                elif r['score'] >= LEVEL_WASPADA:
                    draw_text_outlined(frame, "WASPADA - Tanda Kelelahan", (w//2-150, h//2-50), 0.6, COLOR_WARN, 2)
                elif r['score'] >= LEVEL_HATI_HATI:
                    draw_text_outlined(frame, "HATI-HATI - Mulai Lelah", (w//2-140, h//2-50), 0.6, (0,255,255), 2)

                if r['yawning']:
                    draw_text_outlined(frame, "MENGUAP", (w//2-120, h//2+30), 0.6, (0,200,255), 2)
                if r['head_down_raw']:
                    draw_text_outlined(frame, "NUNDUK", (w//2-100, h//2+60), 0.6, COLOR_DANGER, 2)

        cv2.rectangle(frame, (0,0), (w,35), (15,15,15), -1); draw_text_outlined(frame, f"RPi5 V5 Premium | Subjek {subject_id} | FPS: {int(fps_ema)} | {state}", (10, 23), 0.5, COLOR_INFO, 1)
        cv2.rectangle(frame, (0,h-22), (w,h), (15,15,15), -1); cv2.putText(frame, "q:Exit | c:Calib | r:Reset | s:Rec | f:Flip | h:Graph | u:UI", (10, h-6), FONT, 0.42, COLOR_INFO, 1)
        cv2.imshow('Drowsiness Detection RPi5 V5', frame); key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('r'): state, det = "READY", None
        elif key==ord('s') and state=="MONITORING":
            recording = not recording
            if recording:
                rec_start = t
                last_perf_log = 0.0
                if recorder is None:
                    recorder = VideoRecorder(video_path, (w, h), 20.0)
                    if recorder.start():
                        rec_codec = recorder.codec
                        print(f"[INFO] Recording: {video_path} ({rec_codec})")
                    else:
                        recorder = None
                        rec_codec = None
                        print(f"[ERROR] Gagal membuka VideoWriter: {video_path}")
                if perf_file is None:
                    perf_file = open(perf_path, 'w', newline='')
                    perf_writer = csv.writer(perf_file)
                    perf_writer.writerow(["Time", "Sec", "FPS", "Lat_ms"])
            else:
                if recorder:
                    recorder.stop(); recorder = None; rec_codec = None
                if perf_file:
                    perf_file.close(); perf_file = None; perf_writer = None
        elif key==ord('f'): flip_vertical = not flip_vertical
        elif key==ord('h'): show_g = not show_g

    if recorder: recorder.stop()
    if perf_file: perf_file.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
