"""
dmd_parser.py  —  DMD Parser
Output:
  - events.csv  : yawning/microsleep/drowsiness events dengan timestamp
  - perclos.csv : PERCLOS per detik dihitung dari eye state
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
FPS = 30  # default, sesuaikan jika perlu

def frames_to_timestamp(frame, fps=FPS):
    total_ms = int((frame / fps) * 1000)
    minutes  = total_ms // 60000
    seconds  = (total_ms % 60000) // 1000
    ms       = total_ms % 1000
    # Pakai format MM:SS.mmm (titik sebelum ms) supaya Excel tidak auto-convert
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

# ── Auto-detect JSON ────────────────────────────────────────────────────────
json_files = sorted(SCRIPT_DIR.glob("*.json"))
if not json_files:
    print("[ERROR] Tidak ada file .json di folder ini.")
    exit(1)

# ── Kategori yang kita butuhkan ─────────────────────────────────────────────
# Untuk events.csv → semua action yg mengandung kata ini
FATIGUE_KEYWORDS = [
    "yawning", "microsleep", "drowsy", "sleepy",
    "mouth_state", "fatigue", "vigilance", "nodding"
]
# Untuk PERCLOS → eyes close vs open
EYE_CLOSE_KEYWORDS = ["close", "closing"]
EYE_OPEN_KEYWORDS  = ["open", "opening"]

all_events_rows = []
all_perclos_rows = []

for json_path in json_files:
    print(f"\n[PARSING] {json_path.name}")

    with open(json_path, "r") as f:
        data = json.load(f)

    ol = data["openlabel"]
    parts   = json_path.stem.split("_")
    group   = parts[0] if len(parts) > 0 else "?"
    subject = parts[1] if len(parts) > 1 else "?"
    session = parts[2] if len(parts) > 2 else "?"

    top_actions = ol.get("actions", {})

    # Print semua action types yang ditemukan
    all_types = sorted(set(v.get("type","?") for v in top_actions.values()))
    print(f"  Semua action types ({len(all_types)}):")
    for t in all_types:
        print(f"    - {t}")

    # Hitung total frames
    total_frames = max(
        interval["frame_end"]
        for act_val in top_actions.values()
        for interval in act_val.get("frame_intervals", [])
    ) + 1

    # ── Build events.csv (fatigue + blink events) ──────────────────────────
    for act_val in top_actions.values():
        action_type = act_val.get("type", "unknown").lower()
        
        # Yawning events
        if any(kw in action_type for kw in FATIGUE_KEYWORDS):
            for interval in act_val.get("frame_intervals", []):
                fs = interval["frame_start"]
                fe = interval["frame_end"]
                dur_frames = fe - fs + 1
                dur_sec    = round(dur_frames / FPS, 3)
                all_events_rows.append({
                    "source":       json_path.stem,
                    "group":        group,
                    "subject":      subject,
                    "session":      session,
                    "action_type":  act_val.get("type", "unknown"),
                    "time_start":   frames_to_timestamp(fs),
                    "time_end":     frames_to_timestamp(fe),
                    "duration_sec": dur_sec,
                    "frame_start":  fs,
                    "frame_end":    fe,
                })

        # Blink events — dari label blinks/blinking, filter durasi 2-15 frame
        if action_type == "blinks/blinking":
            for interval in act_val.get("frame_intervals", []):
                fs = interval["frame_start"]
                fe = interval["frame_end"]
                dur_frames = fe - fs + 1
                # Filter: blink valid = 2-15 frame (~67ms-500ms)
                if 2 <= dur_frames <= 15:
                    all_events_rows.append({
                        "source":       json_path.stem,
                        "group":        group,
                        "subject":      subject,
                        "session":      session,
                        "action_type":  "blink",
                        "time_start":   frames_to_timestamp(fs),
                        "time_end":     frames_to_timestamp(fe),
                        "duration_sec": round(dur_frames / FPS, 3),
                        "frame_start":  fs,
                        "frame_end":    fe,
                    })

    # ── Build PERCLOS per detik (standar: backward-looking window, hanya eyes_state/close) ──
    # Tandai tiap frame: 1=eyes STRICTLY closed (eyes_state/close saja)
    eye_strictly_closed = [0] * total_frames

    for act_val in top_actions.values():
        action_type = act_val.get("type", "unknown").lower()
        # Hanya eyes_state/close — TIDAK include opening/closing
        if action_type != "eyes_state/close":
            continue
        for interval in act_val.get("frame_intervals", []):
            for f in range(interval["frame_start"], min(interval["frame_end"]+1, total_frames)):
                eye_strictly_closed[f] = 1

    # PERCLOS standar: % waktu mata tertutup dalam window backward 30-60 detik
    # Denominator = semua frame dalam window (bukan hanya labeled frames)
    window_frames = FPS * 60  # 60 detik window

    for second_idx in range(0, total_frames // FPS):
        center_frame = second_idx * FPS
        # Backward-looking: dari (frame - window) sampai frame sekarang
        win_start = max(0, center_frame - window_frames)
        win_end   = center_frame + 1  # inklusif sampai frame sekarang
        win_len   = win_end - win_start

        closed_count = sum(eye_strictly_closed[win_start:win_end])
        # Denominator = total frame dalam window (BUKAN hanya open+closed)
        perclos = round(closed_count / win_len, 4) if win_len > 0 else None

        all_perclos_rows.append({
            "source":        json_path.stem,
            "group":         group,
            "subject":       subject,
            "session":       session,
            "time":          frames_to_timestamp(center_frame),
            "second":        second_idx,
            "perclos":       perclos,
            "frames_closed": closed_count,
            "window_frames": win_len,
        })

# ── Tulis events terpisah ───────────────────────────────────────────────────
all_events_rows.sort(key=lambda x: x["frame_start"])

yawning_rows = [r for r in all_events_rows if "yawning" in r["action_type"].lower()]
blink_rows   = [r for r in all_events_rows if r["action_type"] == "blink"]

ev_fields = ["source","group","subject","session","action_type",
             "time_start","time_end","duration_sec","frame_start","frame_end"]

# yawning.csv
yawn_path = SCRIPT_DIR / "yawning.csv"
with open(yawn_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ev_fields)
    writer.writeheader()
    writer.writerows(yawning_rows)

# blink.csv
blink_path = SCRIPT_DIR / "blink.csv"
with open(blink_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ev_fields)
    writer.writeheader()
    writer.writerows(blink_rows)

print(f"\n[DONE] yawning.csv → {len(yawning_rows)} yawning events")
print(f"[DONE] blink.csv   → {len(blink_rows)} blink events")

# ── Tulis perclos.csv ────────────────────────────────────────────────────────
perclos_path = SCRIPT_DIR / "perclos.csv"
perclos_fields = ["source","group","subject","session","time","second",
                  "perclos","frames_closed","window_frames"]
with open(perclos_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=perclos_fields)
    writer.writeheader()
    writer.writerows(all_perclos_rows)

print(f"[DONE] perclos.csv → {len(all_perclos_rows)} seconds")

print(f"\n[SUMMARY]")
print(f"  yawning → {len(yawning_rows)}")
print(f"  blink   → {len(blink_rows)}")

if not all_events_rows:
    print("\n[WARNING] events.csv KOSONG — tidak ada action type yang cocok dengan filter.")
    print("          Kemungkinan file JSON ini hanya berisi eye state (bukan yawning/microsleep).")
    print("          Cek apakah ada file JSON lain untuk annotation yawning/drowsiness.")