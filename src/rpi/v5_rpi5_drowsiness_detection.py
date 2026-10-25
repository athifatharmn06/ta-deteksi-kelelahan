import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import math
import subprocess
from collections import deque

# ============================================
# KONFIGURASI SISTEM (LOCKED STABLE V5)
# ============================================
W_EYE = 0.45        
W_MOUTH = 0.25      
W_HEAD = 0.30       

THRESH_EAR_CLOSE = 0.18      
THRESH_MAR_YAWN = 0.35       
THRESH_HEAD_PITCH = 10.0     
TIME_YAWN_MIN_DURATION = 1.8 
TIME_BLINK_MIN_INTERVAL = 0.5
BLINK_COOLDOWN_SEC = 0.7

# MICROSLEEP TUNING
MICROSLEEP_MIN_FRAMES = 30 # Fixed at 1.0s as requested
MICROSLEEP_EAR_FACTOR = 0.50 

# HEAD DROP TUNING
HEAD_DOWN_MIN_SEC = 0.8      
PERCLOS_WARMUP_SEC = 5.0
PERCLOS_GATE_SEC = 2.0

# MAR OSCILLATION (Talking Detection)
MAR_OSC_WINDOW_SEC = 2.0
MAR_OSC_STD_TH = 0.015

# BASELINE MARGINS
MOUTH_BASELINE_MARGIN = 0.10
THRESH_EAR_OPEN = 0.20
MIN_EAR_CLOSE = 0.12
MAX_EAR_CLOSE = 0.22
MIN_MAR_YAWN = 0.35

LEVEL_HATI_HATI = 30
LEVEL_WASPADA   = 50
LEVEL_BAHAYA    = 70

TIME_CONSECUTIVE_FRAMES = 15  
TIME_PERCLOS_WINDOW = 60      
EYE_HOLD_MIN_FRAMES = 6

# UNIVERSAL CONFIGURATION: Relying on Adaptive Calibration for General Use

USE_OSCILLATION_GUARD = True   
USE_PEAK_GATE         = True   
COLOR_ROI    = (255, 255, 0)
COLOR_SAFE   = (0, 255, 0)
COLOR_WARN   = (0, 255, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_INFO   = (255, 255, 255)
COLOR_DEBUG  = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX
W_EYE, W_MOUTH, W_HEAD = 0.45, 0.25, 0.30

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
        # Reduce decay to make accumulator retain fatigue longer
        self.mouth_decay_per_sec = 0.08
        self.head_decay_per_sec = 0.10
        self.eye_decay_per_sec = 0.06
        self.microsleep_decay_per_sec = 0.025
        # Increase event gains so repeated events accumulate more
        self.mouth_event_gain = 18.0
        self.head_event_gain = 16.0
        self.microsleep_event_gain = 30.0
        self.head_hold_gain_per_sec = 3.0
        self.eye_hold_gain_per_sec = 2.6
        self.perclos_gain_per_sec = 0.9
        self.microsleep_hold_gain_per_sec = 8.0
        self.recent_yawns = deque(); self.recent_head_drops = deque(); self.recent_microsleeps = deque()
        self.recent_window_sec = 600.0
        # Persistent counters
        self.microsleep_count = 0
        self.yawn_count = 0
        self.head_drop_count = 0

    def update(self, dt, now_ts, h_down, perclos, micro):
        scale = 0.55 if (h_down or perclos > 20.0 or micro) else 1.0
        self.mouth_acc = max(0, self.mouth_acc - self.mouth_decay_per_sec * dt)
        self.head_acc = max(0, self.head_acc - self.head_decay_per_sec * dt * scale)
        self.eye_acc = max(0, self.eye_acc - self.eye_decay_per_sec * dt * scale)
        self.microsleep_acc = max(0, self.microsleep_acc - self.microsleep_decay_per_sec * dt * scale)
        if h_down: self.head_acc = min(100, self.head_acc + self.head_hold_gain_per_sec * dt)
        if perclos > 20.0: self.eye_acc = min(100, self.eye_acc + self.perclos_gain_per_sec * dt * (perclos/20.0))
        if micro: self.microsleep_acc = min(100, self.microsleep_acc + self.microsleep_hold_gain_per_sec * dt)
        for d in [self.recent_yawns, self.recent_head_drops, self.recent_microsleeps]:
            while d and (now_ts - d[0]) > self.recent_window_sec: d.popleft()

    def add_yawn(self, ts):
        self.mouth_acc = min(100, self.mouth_acc + self.mouth_event_gain)
        self.recent_yawns.append(ts)
        self.yawn_count += 1
    def add_head_drop(self, ts):
        self.head_acc = min(100, self.head_acc + self.head_event_gain)
        self.recent_head_drops.append(ts)
        self.head_drop_count += 1
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
    def __init__(self, fps=30, baseline_pitch=0.0, baseline_ear=0.30, baseline_mouth=0.0, baseline_mw=0.0):
        self.fps = fps
        self.bp = baseline_pitch
        self.be = baseline_ear
        self.bm = baseline_mouth
        self.bmw = baseline_mw
        self.eb = deque(maxlen=5)
        self.mb, self.pb = deque(maxlen=3), deque(maxlen=3)
        self.ecf = self.ecf_gap = self.hdf = self.vbc = self.ecf_blink = 0
        self.blink_active = False
        self.ys = 'IDLE'; self.yhf = 0; self.ypm = 0.0; self.lyt = 0.0
        self.mct = deque(); self.map = False
        self.perclos_calc = PERCLOSCalculator(60, fps); self.acc = DrowsinessAccumulator()
        self.bp = baseline_pitch
        self.start_ts = None
        self.lut = None
        self.be = baseline_ear
        self.bm = baseline_mouth
        self.ms = None; self.pm = self.ph = False
        self.last_blink_ts = 0.0
        self.blink_rearm_ts = 0.0
        self.monitor_start_ts = None
        self.perclos_high_start = None
        self.mar_hist = deque(maxlen=max(5, int(MAR_OSC_WINDOW_SEC * fps)))
        self.head_down_start = None
        self.th_ear_close = THRESH_EAR_CLOSE
        self.th_ear_open = THRESH_EAR_OPEN
        self.th_mar_yawn = THRESH_MAR_YAWN

    def update(self, ear, mar, pitch, fps, t=None, mw_ratio=0.0):
        if t is None: t = time.time()
        
        self.eb.append(ear)
        es = float(np.median(self.eb)) if len(self.eb) >= 3 else ear
        ms, ps = mar, pitch
        
        # PERCLOS calculation (Now enabled during yawning as requested)
        p_data = self.perclos_calc.update(es, self.th_ear_close, skip=False)
        # Closure Detection
        ec = (es < self.th_ear_close)
        # Blink detection (Always active even during yawning)
        if ec:
            self.ecf_blink += 1
            if self.ecf_blink >= 2 and not self.blink_active: # Require at least 2 frames
                if t >= self.blink_rearm_ts and (t - self.last_blink_ts) >= TIME_BLINK_MIN_INTERVAL:
                    self.vbc += 1
                    self.last_blink_ts = t
                    self.blink_rearm_ts = t + BLINK_COOLDOWN_SEC
                self.blink_active = True
        elif es > self.th_ear_open:
            self.ecf_blink = 0
            if self.blink_active:
                self.blink_active = False
            
        is_yawning = self.ys != 'IDLE' # Used to suppress microsleep but allow PERCLOS
        pd = ps - self.bp
        hp_th = THRESH_HEAD_PITCH
        # Sensitive HeadDrop: Using journal-based 10.0 deg deviation
        hdr = pd < -hp_th
        hu = pd > hp_th and ps > 3.0
        mouth_open_th = max(THRESH_MAR_YAWN, self.bm + MOUTH_BASELINE_MARGIN)
        m_ab = ms > mouth_open_th
        if self.monitor_start_ts is None: self.monitor_start_ts = t
        warmup_active = (t - self.monitor_start_ts) < PERCLOS_WARMUP_SEC
        self.mar_hist.append(ms)
        mar_std = float(np.std(self.mar_hist)) if len(self.mar_hist) >= 5 else 0.0
        if m_ab and not self.map: self.mct.append(t)
        while self.mct and (t - self.mct[0]) > 4.0: self.mct.popleft()
        is_t = (len(self.mct) > 3) or (mar_std > MAR_OSC_STD_TH and ms < (mouth_open_th + 0.10))
        self.map = m_ab
        # 3. Yawn State Machine (Adaptive Peak Detection with Smile Guard)
        is_smiling = mw_ratio > (self.bmw * 1.30) if self.bmw > 0 else False
        m_ab_yawn = ms >= (self.bm + 0.15) # Proven starter for 82% F1
        
        if self.ys == 'IDLE' and m_ab_yawn and not is_smiling: 
            self.ys, self.yhf, self.ypm = 'BUILDING', 1, ms
        elif self.ys == 'BUILDING':
            if not m_ab_yawn: 
                self.ys, self.yhf = 'IDLE', 0
            else:
                self.yhf += 1; self.ypm = max(self.ypm, ms)
                peak_min = self.bm + 0.12 # Proven peak for 82% F1
                ear_squint = (ear < (self.be * 0.95)) or (self.ypm > 0.60)
                
                if is_smiling: self.ys, self.yhf = 'IDLE', 0
                
                if self.yhf >= int(TIME_YAWN_MIN_DURATION * fps) and self.ypm >= peak_min and ear_squint:
                    self.ys, self.lyt = 'CONFIRMED', t; self.acc.add_yawn(t)
        elif self.ys == 'CONFIRMED' and not m_ab_yawn: self.ys = 'COOLDOWN'
        elif self.ys == 'COOLDOWN' and (t - self.lyt) >= 5.0: self.ys = 'IDLE'
        
        # 4. Microsleep Logic
        m_det = hd_det = False
        if warmup_active:
            self.ecf = 0
            self.ms = None
        else:
            # Use strict closure logic with Noise Suppression and Mouth Guard
            ec_micro = es < (self.th_ear_close * MICROSLEEP_EAR_FACTOR)
            # Block microsleep if subject is yawning OR talking/smiling (Mouth MAR is active)
            if ec_micro and not is_yawning and ms <= (self.bm + 0.05): 
                self.ecf += 1
            else: 
                self.ecf = 0
                
            if self.ecf >= MICROSLEEP_MIN_FRAMES:
                if self.ms is None: self.ms = t
                m_det = True
            else: 
                self.ms = None
        if hdr:
            if self.head_down_start is None:
                self.head_down_start = t
        else:
            self.head_down_start = None

        hdr_guarded = hdr and self.head_down_start is not None and (t - self.head_down_start) >= HEAD_DOWN_MIN_SEC

        if hdr_guarded: self.hdf = min(self.hdf+1, 60)
        else: self.hdf = max(self.hdf-1, 0)
        hd_det = self.hdf >= int(HEAD_DOWN_MIN_SEC * fps)
        if self.lut is None: self.lut = t
        dt = min(t - self.lut, 0.5); self.lut = t
        if m_det and not self.pm: self.acc.add_microsleep(t)
        if hd_det and not self.ph: self.acc.add_head_drop(t)
        self.pm, self.ph = m_det, hd_det
        if p_data['perclos'] > 20.0:
            if self.perclos_high_start is None:
                self.perclos_high_start = t
        else:
            self.perclos_high_start = None

        allow_perclos = (self.perclos_high_start is not None
                         and (t - self.perclos_high_start) >= PERCLOS_GATE_SEC
                         and not warmup_active)
        self.acc.update(dt, t, hdr_guarded, p_data['perclos'] if allow_perclos else 0.0, m_det)
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
        return {'ear':es,'mar':ms,'pitch':ps,'pitch_deviation':pd,'perclos':p_data['perclos'],'eye_closed':ec,'microsleep':m_det,'yawning':self.ys=='CONFIRMED','head_down':hd_det,'head_down_raw':hdr_guarded,'head_up':hu,'score':fs,'vbc':self.vbc,'yawn_count':len(self.acc.recent_yawns),'micro_count':len(self.acc.recent_microsleeps),'microsleep_count':self.acc.microsleep_count,'yawn_total':self.acc.yawn_count,'head_drop_total':self.acc.head_drop_count,'yawn_state':self.ys,'is_talking':is_t,'mouth_acc':self.acc.mouth_acc,'head_acc':self.acc.head_acc,'eye_acc':self.acc.eye_acc,'microsleep_acc':self.acc.microsleep_acc,'inst':ins,'acc':ascore,'ecf':self.ecf}

class PERCLOSCalculator:
    def __init__(self, ws=60, fps=30):
        self.ws = int(ws*fps); self.es = deque(maxlen=self.ws); self.bc = 0; self.lbt = time.time(); self.br = 0; self.les = False
    def update(self, ear, th, skip=False):
        if skip: return {'perclos': (sum(self.es)/len(self.es)*100) if len(self.es)>0 else 0, 'blink_count': self.bc}
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
    state = "READY"; det, bv = None, {}; fps_ema, pt = 30.0, time.time()
    flip_vertical = False; show_g = True; compact = False
    baseline_values = {'pitch': 0.0}
    eh, mh, ph = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
    c_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        raw = frame.copy()
        if flip_vertical:
            frame = cv2.flip(frame, 0)
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
            def draw_tracking():
                eye_idx = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
                mouth_idx = [61, 291, 82, 87, 13, 14, 312, 317]
                for idx in eye_idx:
                    x, y = get_p(idx)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                for idx in mouth_idx:
                    x, y = get_p(idx)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                p10, p152, p4 = get_p(10), get_p(152), get_p(4)
                cv2.line(frame, p10, p152, (255, 0, 0), 1)
                cv2.circle(frame, p4, 2, (255, 0, 0), -1)
            
            ear, mar, pose = calc_ear(), calc_mar(), calc_pose()
            eh.append(ear); mh.append(mar); ph.append(pose)
            draw_tracking()
            
            xc, yc = [l.x for l in lm], [l.y for l in lm]
            x1, y1, x2, y2 = int(min(xc)*w)-15, int(min(yc)*h)-15, int(max(xc)*w)+15, int(max(yc)*h)+15
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_ROI, 1)
            roi = raw[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

            if state == "READY":
                # Simplified READY UI (No bulky box)
                draw_text_outlined(frame, "STATUS: SYSTEM READY", (w//2-100, h-80), 0.6, COLOR_SAFE, 2)
                draw_text_outlined(frame, "Press 'C' to start 3s Calibration", (w//2-140, h-55), 0.5, COLOR_INFO, 1)
                
                # Small real-time stats at bottom left instead of center
                rt_y = h - 140
                cv2.putText(frame, f"EAR: {ear:.3f} | MAR: {mar:.3f} | Pitch: {pose:.1f}", (20, rt_y), FONT, 0.50, COLOR_DEBUG, 1, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {int(fps_ema)}", (20, rt_y+20), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
            elif state == "CALIBRATING":
                elapsed_calibration = time.time() - c_start
                countdown = max(0, int(CALIB_OPEN_SEC - elapsed_calibration))
                cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), (30, 30, 30), -1)
                cv2.rectangle(frame, (w//2-180, h//2-100), (w//2+180, h//2+100), COLOR_SAFE, 2)
                cv2.putText(frame, "KALIBRASI BERJALAN", (w//2-70, h//2-70), FONT, 0.65, COLOR_SAFE, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Sisa: {countdown} detik", (w//2-50, h//2-30), FONT, 0.65, COLOR_INFO, 2, cv2.LINE_AA)
                progress_filled = int((elapsed_calibration / CALIB_OPEN_SEC) * 200)
                cv2.rectangle(frame, (w//2-100, h//2+10), (w//2-100 + progress_filled, h//2+20), COLOR_SAFE, -1)
                c_data.append({'e':ear, 'm':mar, 'p':pose})
                cv2.putText(frame, f"Sampel: {len(c_data)} frame", (w//2-90, h//2+40), FONT, 0.50, COLOR_INFO, 1, cv2.LINE_AA)
                if elapsed_calibration >= CALIB_OPEN_SEC and len(c_data) > 0:
                    e_samples = [x['e'] for x in c_data]
                    be = np.mean(e_samples)
                    se = np.std(e_samples)
                    bm = np.mean([x['m'] for x in c_data])
                    bp = np.median([x['p'] for x in c_data])
                    
                    global THRESH_EAR_CLOSE, THRESH_EAR_OPEN, THRESH_MAR_YAWN
                    # Statistical Adaptive Thresholding (Ref: IET Research / ResearchGate)
                    # Statistical Adaptive Thresholding (Three-Sigma Rule: 3.0)
                    sigma_cap = be * 0.15
                    se_guarded = max(min(se, sigma_cap), 0.01) 
                    THRESH_EAR_CLOSE = min(MAX_EAR_CLOSE, be - 3.0 * se_guarded)
                    THRESH_EAR_OPEN = min(MAX_EAR_CLOSE + 0.05, max(THRESH_EAR_CLOSE + 0.01, be - se_guarded))
                    
                    # Offset for stable baseline adaptation
                    THRESH_MAR_YAWN = max(MIN_MAR_YAWN, bm + 0.12)
                    det = DrowsinessDetector(fps=int(fps_ema), baseline_pitch=bp, baseline_ear=be, baseline_mouth=bm)
                    det.lut = time.time()
                    bv = {'p': bp}
                    baseline_values['pitch'] = bp
                    state = "MONITORING"
            elif state=="MONITORING" and det:
                r = det.update(ear, mar, pose, int(fps_ema))
                # normalize some keys expected by v4 UI
                r['total_yawns'] = r.get('yawn_total', r.get('yawn_count', 0))
                r['valid_blink_count'] = r.get('vbc', 0)
                r['blink_count'] = r.get('vbc', 0)
                r['yawn_peak_mar'] = r.get('ypm', r.get('yawn_peak_mar', 0.0))
                r['yawn_hold_frames'] = r.get('yhf', r.get('yawn_hold_frames', 0))
                
                # HUD/UI adapted from v4 (UI/UX only)
                # map compact/show flags to v4 names
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

                # left panel lines (v4 style)
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
                        (f"Talk: {'YA' if r.get('is_talking',False) else 'TIDAK'}", COLOR_INFO),
                        ("", COLOR_INFO),
                        ("=== PERCLOS ===", COLOR_INFO),
                        (f"PERCLOS: {r['perclos']:.1f}%", COLOR_WARN if r['perclos']>20 else COLOR_SAFE),
                        ("", COLOR_INFO),
                        ("=== COUNTERS ===", COLOR_INFO),
                        (f"Blink: {r.get('blink_count',0)}", COLOR_INFO),
                        (f"Yawns: {r.get('total_yawns',0)}", COLOR_INFO),
                        (f"HeadDn: {r.get('head_drop_total',0)}", COLOR_INFO),
                        (f"Microsleep: {r.get('microsleep_count',0)}", COLOR_INFO),
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

                # RIGHT PANEL (status + ROI + graphs)
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

                # ROI preview
                roi_y = status_y + 84
                roi_size = min(100, status_w - 10) if not ui_compact_mode else min(76, status_w - 10)
                if roi is not None and roi.size:
                    roi_resized = cv2.resize(roi, (roi_size, roi_size))
                    roi_x = status_x + (status_w - roi_size) // 2
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), COLOR_ROI, 1)
                    frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size] = roi_resized
                    cv2.putText(frame, "ROI", (roi_x, roi_y - 6), FONT, 0.40, COLOR_ROI, 1, cv2.LINE_AA)

                # graphs
                graph_y_start = roi_y + roi_size + 38
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

                # Center warnings (outlined)
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

        cv2.rectangle(frame, (0,0), (w,35), (15,15,15), -1); draw_text_outlined(frame, f"RPi5 V5 Premium | FPS: {int(fps_ema)} | {state}", (10, 23), 0.5, COLOR_INFO, 1)
        cv2.rectangle(frame, (0,h-22), (w,h), (15,15,15), -1); cv2.putText(frame, "q:Exit | c:Calib | r:Reset | f:Flip | h:Graph | u:UI", (10, h-6), FONT, 0.42, COLOR_INFO, 1)
        cv2.imshow('Drowsiness Detection RPi5 V5', frame); key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('r'): state, det = "READY", None
        elif key==ord('f'): flip_vertical = not flip_vertical
        elif key==ord('h'): show_g = not show_g
        elif key==ord('c'):
            state = "CALIBRATING"
            c_start = time.time()
            c_data = []

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
