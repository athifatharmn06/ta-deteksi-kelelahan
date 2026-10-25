import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np


THRESH_MAR_YAWN = 0.20
TIME_YAWN_MIN_DURATION = 1.5
USE_PEAK_GATE = True
USE_OSCILLATION_GUARD = True


@dataclass
class FrameMeta:
    frame_idx: int
    mar: float
    state: str
    is_talking: bool


class YawnStateTracker:
    def __init__(self, fps: float):
        self.fps = max(fps, 1.0)
        self._yawn_state = "IDLE"
        self.yawn_hold_frames = 0
        self.yawn_peak_mar = 0.0
        self.last_yawn_time = 0.0
        self.yawn_cooldown = 5.0

        self._mar_above_prev = False
        self._mar_cross_times = []
        self._osc_window = 4.0
        self._osc_max = 3

    def update(self, mar_smooth: float, t_sec: float):
        mar_threshold = THRESH_MAR_YAWN
        peak_gate_abs = max(mar_threshold * 1.6, 0.50)
        mar_above_now = mar_smooth > mar_threshold

        if mar_above_now and not self._mar_above_prev:
            self._mar_cross_times.append(t_sec)

        self._mar_cross_times = [x for x in self._mar_cross_times if (t_sec - x) <= self._osc_window]
        is_talking = len(self._mar_cross_times) > self._osc_max if USE_OSCILLATION_GUARD else False
        self._mar_above_prev = mar_above_now

        if self._yawn_state == "IDLE":
            if mar_above_now and not is_talking:
                self._yawn_state = "BUILDING"
                self.yawn_hold_frames = 1
                self.yawn_peak_mar = mar_smooth

        elif self._yawn_state == "BUILDING":
            if (not mar_above_now) or is_talking:
                self._yawn_state = "IDLE"
                self.yawn_hold_frames = 0
                self.yawn_peak_mar = 0.0
            else:
                self.yawn_hold_frames += 1
                self.yawn_peak_mar = max(self.yawn_peak_mar, mar_smooth)

                min_hold = int(TIME_YAWN_MIN_DURATION * self.fps)
                peak_condition = (self.yawn_peak_mar >= peak_gate_abs) if USE_PEAK_GATE else True
                if self.yawn_hold_frames >= min_hold and peak_condition:
                    self._yawn_state = "CONFIRMED"
                    self.last_yawn_time = t_sec

        elif self._yawn_state == "CONFIRMED":
            if not mar_above_now:
                self._yawn_state = "COOLDOWN"
                self.yawn_hold_frames = 0
                self.yawn_peak_mar = 0.0

        elif self._yawn_state == "COOLDOWN":
            if (t_sec - self.last_yawn_time) >= self.yawn_cooldown:
                self._yawn_state = "IDLE"

        return self._yawn_state, is_talking


def euclidean_distance(p1, p2):
    return float(np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)))


def get_landmark_point(landmarks, index, width, height):
    return int(landmarks[index].x * width), int(landmarks[index].y * height)


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


def pick_yawn_frames(meta, fps):
    states = [m.state for m in meta]
    n = len(meta)

    conf_start = None
    for i in range(1, n):
        if states[i] == "CONFIRMED" and states[i - 1] != "CONFIRMED":
            conf_start = i
            break

    if conf_start is None:
        raise RuntimeError("No CONFIRMED state found in yawning video.")

    b_end = conf_start - 1
    b_start = b_end
    while b_start >= 0 and states[b_start] == "BUILDING":
        b_start -= 1
    b_start += 1

    if b_start >= conf_start:
        raise RuntimeError("No BUILDING segment found before CONFIRMED.")

    pre = max(0, b_start - int(0.4 * fps))
    b_mid = b_start + (conf_start - b_start) // 2
    c_mid = min(n - 1, conf_start + int(0.6 * fps))

    picks = [pre, b_start, b_mid, conf_start, c_mid]
    return picks


def pick_talking_frames(meta, fps):
    n = len(meta)
    states = [m.state for m in meta]
    talks = [m.is_talking for m in meta]

    for i in range(1, n):
        if states[i - 1] == "BUILDING" and states[i] == "IDLE" and talks[i]:
            build_end = i - 1
            build_start = build_end
            while build_start >= 0 and states[build_start] == "BUILDING":
                build_start -= 1
            build_start += 1

            pre = max(0, build_start - int(0.4 * fps))
            rise = build_start
            reset = i
            post = min(n - 1, i + int(0.5 * fps))

            seg_start = max(build_start, build_start - int(0.2 * fps))
            seg_end = min(build_end, build_start + int(0.8 * fps))
            if seg_end < seg_start:
                peak_idx = build_start
            else:
                peak_idx = max(range(seg_start, seg_end + 1), key=lambda k: meta[k].mar)

            return [pre, rise, peak_idx, reset, post]

    talking_indices = [i for i, t in enumerate(talks) if t]
    if not talking_indices:
        center = n // 2
    else:
        center = talking_indices[len(talking_indices) // 2]

    return [
        max(0, center - int(1.0 * fps)),
        max(0, center - int(0.5 * fps)),
        center,
        min(n - 1, center + int(0.5 * fps)),
        min(n - 1, center + int(1.0 * fps)),
    ]


def analyze_video(video_path, face_mesh):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    mar_buffer = []
    tracker = YawnStateTracker(fps)
    metadata = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            mar = calculate_mar(lm, w, h)
            mar_buffer.append(mar)
            if len(mar_buffer) > 5:
                mar_buffer.pop(0)
            mar_smooth = float(np.mean(mar_buffer))

            state, is_talking = tracker.update(mar_smooth, frame_idx / fps)
            metadata.append(FrameMeta(frame_idx, mar_smooth, state, is_talking))

        frame_idx += 1

    cap.release()

    if not metadata:
        raise RuntimeError(f"No face landmarks found in video: {video_path}")

    return metadata, fps


def extract_annotated_face_crop(video_path, frame_idx, state, mar, is_talking, face_mesh, mp_drawing, mp_styles):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_offsets = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]

    frame = None
    face_landmarks = None
    used_frame_idx = frame_idx

    # Some candidate frames may miss landmarks due blur/occlusion; search nearby.
    for off in search_offsets:
        cand = frame_idx + off
        if cand < 0 or cand >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, cand)
        ok, candidate_frame = cap.read()
        if not ok:
            continue

        h, w = candidate_frame.shape[:2]
        rgb = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            frame = candidate_frame
            face_landmarks = result.multi_face_landmarks[0]
            used_frame_idx = cand
            break

    cap.release()

    if frame is None or face_landmarks is None:
        raise RuntimeError(f"No face landmarks near frame {frame_idx}")

    h, w = frame.shape[:2]

    annotated = frame.copy()
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
    )

    xs = [int(p.x * w) for p in face_landmarks.landmark]
    ys = [int(p.y * h) for p in face_landmarks.landmark]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(0.30 * bw)
    pad_y = int(0.35 * bh)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

    label = f"F{used_frame_idx} | MAR={mar:.3f} | {state} | talk={int(is_talking)}"
    cv2.rectangle(annotated, (cx1, max(0, cy1 - 26)), (min(w - 1, cx1 + 420), cy1), (10, 10, 10), -1)
    cv2.putText(
        annotated,
        label,
        (cx1 + 5, max(14, cy1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    crop = annotated[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop at frame {frame_idx}")

    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def compose_figure(yawn_imgs, talk_imgs, out_png, out_pdf):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8.8))

    for i in range(5):
        axes[0, i].imshow(yawn_imgs[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(talk_imgs[i])
        axes[1, i].axis("off")

    fig.text(0.5, 0.96, "Row 1: Successful Yawn Detection (BUILDING -> CONFIRMED)", ha="center", va="top", fontsize=16)
    fig.text(0.5, 0.515, "Row 2: Talking Suppressed by Oscillation Guard (IDLE/Reset, no CONFIRMED)", ha="center", va="top", fontsize=16)


    plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.92], h_pad=3.2)
    plt.subplots_adjust(hspace=0.2)
    fig.savefig(out_png, dpi=400)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative 2x5 figure for yawn vs talking suppression.")
    parser.add_argument(
        "--yawn_video",
        default=r"DataSet\YawDD\Mirror\Mirror\Male_mirror Avi Videos\6-MaleNoGlasses-Yawning.avi",
    )
    parser.add_argument(
        "--talk_video",
        default=r"DataSet\YawDD\Mirror\Mirror\Male_mirror Avi Videos\6-MaleNoGlasses-Talking.avi",
    )
    parser.add_argument(
        "--out_png",
        default=r"Jurnal JNTETI\Figures\qualitative_visual_results_subject6.png",
    )
    parser.add_argument(
        "--out_pdf",
        default=r"Jurnal JNTETI\Figures\qualitative_visual_results_subject6.pdf",
    )
    parser.add_argument(
        "--out_meta",
        default=r"Jurnal JNTETI\Figures\qualitative_visual_results_subject6_frames.json",
    )
    args = parser.parse_args()

    yawn_video = Path(args.yawn_video)
    talk_video = Path(args.talk_video)
    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf)
    out_meta = Path(args.out_meta)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        yawn_meta, yawn_fps = analyze_video(yawn_video, face_mesh)
        talk_meta, talk_fps = analyze_video(talk_video, face_mesh)

        yawn_picks = pick_yawn_frames(yawn_meta, yawn_fps)
        talk_picks = pick_talking_frames(talk_meta, talk_fps)

        yawn_imgs = []
        for idx in yawn_picks:
            m = yawn_meta[idx]
            yawn_imgs.append(
                extract_annotated_face_crop(yawn_video, m.frame_idx, m.state, m.mar, m.is_talking, face_mesh, mp_drawing, mp_styles)
            )

        talk_imgs = []
        for idx in talk_picks:
            m = talk_meta[idx]
            talk_imgs.append(
                extract_annotated_face_crop(talk_video, m.frame_idx, m.state, m.mar, m.is_talking, face_mesh, mp_drawing, mp_styles)
            )

    compose_figure(yawn_imgs, talk_imgs, out_png, out_pdf)

    payload = {
        "yawn_video": str(yawn_video),
        "talk_video": str(talk_video),
        "yawn_selected": [
            {
                "meta_index": int(i),
                "frame_idx": int(yawn_meta[i].frame_idx),
                "state": yawn_meta[i].state,
                "mar": float(yawn_meta[i].mar),
                "is_talking": bool(yawn_meta[i].is_talking),
            }
            for i in yawn_picks
        ],
        "talk_selected": [
            {
                "meta_index": int(i),
                "frame_idx": int(talk_meta[i].frame_idx),
                "state": talk_meta[i].state,
                "mar": float(talk_meta[i].mar),
                "is_talking": bool(talk_meta[i].is_talking),
            }
            for i in talk_picks
        ],
        "config": {
            "threshold_mar_yawn": THRESH_MAR_YAWN,
            "time_yawn_min_duration": TIME_YAWN_MIN_DURATION,
            "use_peak_gate": USE_PEAK_GATE,
            "use_oscillation_guard": USE_OSCILLATION_GUARD,
        },
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_meta}")


if __name__ == "__main__":
    main()
