"""
gt_label_builder.py
Ground Truth Label Builder untuk Validasi Threshold Level Kelelahan

Membaca anotasi DMD (*_rgb_ann_drowsiness.json) dan menghasilkan GT label
per segmen 10 detik untuk setiap video (label 0-3).

Label Mapping (prioritas berurutan Bahaya → Waspada → Hati-hati → Aman):
  3 = Bahaya    : eyes_state/close kumulatif >= 1.5 detik dalam segmen
  2 = Waspada   : yawning/without_hand ATAU PERCLOS lokal segmen >= 15%
  1 = Hati-hati : yawning/with_hand ATAU PERCLOS lokal segmen 5-15%
  0 = Aman      : tidak ada kondisi di atas

Usage:
    python gt_label_builder.py                  # semua 15 video
    python gt_label_builder.py --ds 1           # hanya DMD Dataset 1
    python gt_label_builder.py --segment 10     # ukuran segmen (detik)
"""

import json
import os
import sys
import argparse
from pathlib import Path

# ── Konfigurasi ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATASET_BASE = SCRIPT_DIR.parent
RESULTS_DIR  = SCRIPT_DIR.parent.parent / "Jurnal JNTETI" / "Results" / "level_validation" / "gt"

FPS = 30

DS_MAPPING = {i: f"DMD Dataset {i}" for i in range(1, 16)}

# Threshold GT (bisa di-override via argparse)
GT_MICROSLEEP_FRAMES     = int(1.5 * FPS)   # 45 frame: eye close >= 1.5 detik = Bahaya
GT_PERCLOS_WASPADA       = 15.0             # % PERCLOS lokal >= 15% = Waspada
GT_PERCLOS_HATI_HATI_LO  = 5.0             # % PERCLOS lokal >= 5% = Hati-hati
GT_PERCLOS_HATI_HATI_HI  = 15.0


def load_dmd_annotations(json_path):
    """
    Membaca JSON anotasi DMD dan mengekstrak:
    - close_intervals: list (frame_start, frame_end) untuk eyes_state/close
    - yawn_without:    list (frame_start, frame_end) untuk yawning/Yawning without hand
    - yawn_with:       list (frame_start, frame_end) untuk yawning/Yawning with hand
    - total_frames:    jumlah total frame video
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    actions = data['openlabel'].get('actions', {})

    close_intervals  = []
    yawn_without     = []
    yawn_with        = []

    for act_val in actions.values():
        t = act_val.get('type', '').lower()
        intervals = act_val.get('frame_intervals', [])

        if t == 'eyes_state/close':
            for iv in intervals:
                close_intervals.append((iv['frame_start'], iv['frame_end']))
        elif 'yawning without' in t or 'without hand' in t:
            for iv in intervals:
                yawn_without.append((iv['frame_start'], iv['frame_end']))
        elif 'yawning with' in t or 'with hand' in t:
            for iv in intervals:
                yawn_with.append((iv['frame_start'], iv['frame_end']))

    # Total frame = max frame_end + 1
    all_ends = [
        iv['frame_end']
        for act_val in actions.values()
        for iv in act_val.get('frame_intervals', [])
    ]
    total_frames = max(all_ends) + 1 if all_ends else 0

    return close_intervals, yawn_without, yawn_with, total_frames


def compute_gt_labels(close_intervals, yawn_without, yawn_with, total_frames,
                      segment_sec=10):
    """
    Hitung GT label per segmen untuk satu video.

    Returns:
        segments: list of dict {
            'seg_idx': int,
            'frame_start': int,
            'frame_end': int,
            'label': int (0-3),
            'label_name': str,
            'closed_frames': int,
            'perclos_pct': float,
            'has_yawn_without': bool,
            'has_yawn_with': bool
        }
    """
    seg_frames = segment_sec * FPS
    n_segments = max(1, (total_frames + seg_frames - 1) // seg_frames)

    segments = []

    for seg_idx in range(n_segments):
        seg_start = seg_idx * seg_frames
        seg_end   = min(seg_start + seg_frames - 1, total_frames - 1)
        seg_len   = seg_end - seg_start + 1

        # ── 1. Hitung kumulatif frame eyes_state/close dalam segmen ──────────
        closed_frames = 0
        for (fs, fe) in close_intervals:
            overlap_start = max(fs, seg_start)
            overlap_end   = min(fe, seg_end)
            if overlap_end >= overlap_start:
                closed_frames += overlap_end - overlap_start + 1

        perclos_pct = (closed_frames / seg_len * 100.0) if seg_len > 0 else 0.0

        # ── 2. Cek yawning events dalam segmen ───────────────────────────────
        has_yawn_without = any(
            max(fs, seg_start) <= min(fe, seg_end)
            for (fs, fe) in yawn_without
        )
        has_yawn_with = any(
            max(fs, seg_start) <= min(fe, seg_end)
            for (fs, fe) in yawn_with
        )

        # ── 3. Assign label (prioritas: 3 > 2 > 1 > 0) ──────────────────────
        if closed_frames >= GT_MICROSLEEP_FRAMES:
            label = 3
        elif has_yawn_without or perclos_pct >= GT_PERCLOS_WASPADA:
            label = 2
        elif has_yawn_with or (GT_PERCLOS_HATI_HATI_LO <= perclos_pct < GT_PERCLOS_HATI_HATI_HI):
            label = 1
        else:
            label = 0

        label_names = {0: 'Aman', 1: 'Hati-hati', 2: 'Waspada', 3: 'Bahaya'}

        segments.append({
            'seg_idx'        : seg_idx,
            'frame_start'    : int(seg_start),
            'frame_end'      : int(seg_end),
            'label'          : label,
            'label_name'     : label_names[label],
            'closed_frames'  : int(closed_frames),
            'perclos_pct'    : round(perclos_pct, 3),
            'has_yawn_without': bool(has_yawn_without),
            'has_yawn_with'  : bool(has_yawn_with),
        })

    return segments


def process_dataset(ds_id, segment_sec=10, verbose=True):
    """Proses satu dataset, simpan GT ke JSON. Return summary dict."""
    ds_folder = DATASET_BASE / DS_MAPPING[ds_id]

    # Cari JSON anotasi
    json_files = list(ds_folder.glob('*_rgb_ann_drowsiness.json'))
    if not json_files:
        json_files = list(ds_folder.glob('*.json'))
    if not json_files:
        print(f"[WARN] DS{ds_id:02d}: Tidak ada JSON anotasi di {ds_folder}")
        return None

    json_path = json_files[0]

    try:
        close_ivs, yawn_wo, yawn_wi, total_frames = load_dmd_annotations(json_path)
    except Exception as e:
        print(f"[ERROR] DS{ds_id:02d}: Gagal load JSON: {e}")
        return None

    segments = compute_gt_labels(close_ivs, yawn_wo, yawn_wi, total_frames,
                                  segment_sec=segment_sec)

    # Hitung distribusi label
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    for seg in segments:
        dist[seg['label']] += 1

    summary = {
        'ds_id'       : ds_id,
        'json_file'   : json_path.name,
        'total_frames': total_frames,
        'total_seconds': round(total_frames / FPS, 1),
        'n_segments'  : len(segments),
        'segment_sec' : segment_sec,
        'label_dist'  : dist,
        'segments'    : segments,
        'gt_config': {
            'GT_MICROSLEEP_FRAMES'   : GT_MICROSLEEP_FRAMES,
            'GT_PERCLOS_WASPADA'     : GT_PERCLOS_WASPADA,
            'GT_PERCLOS_HATI_HATI_LO': GT_PERCLOS_HATI_HATI_LO,
            'GT_PERCLOS_HATI_HATI_HI': GT_PERCLOS_HATI_HATI_HI,
        }
    }

    if verbose:
        print(f"  DS{ds_id:02d} | {len(segments)} segmen | "
              f"Aman={dist[0]} Hati-hati={dist[1]} Waspada={dist[2]} Bahaya={dist[3]}")

    # Simpan
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"gt_labels_ds{ds_id:02d}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="GT Label Builder untuk Validasi Level Kelelahan")
    parser.add_argument('--ds', type=int, default=None, help="Hanya proses dataset ID tertentu (1-15)")
    parser.add_argument('--segment', type=int, default=10, help="Ukuran segmen dalam detik (default: 10)")
    args = parser.parse_args()

    print("=" * 65)
    print("GT LABEL BUILDER — Validasi Threshold Level Kelelahan")
    print(f"Segment size: {args.segment} detik | FPS: {FPS}")
    print(f"Threshold Bahaya  : eyes_state/close >= {GT_MICROSLEEP_FRAMES} frame = {GT_MICROSLEEP_FRAMES/FPS}s")
    print(f"Threshold Waspada : yawning_without_hand OR PERCLOS >= {GT_PERCLOS_WASPADA}%")
    print(f"Threshold Hati-hati: yawning_with_hand OR PERCLOS {GT_PERCLOS_HATI_HATI_LO}-{GT_PERCLOS_HATI_HATI_HI}%")
    print("=" * 65)

    ds_ids = [args.ds] if args.ds else list(range(1, 16))

    all_summaries = []
    total_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    for ds_id in ds_ids:
        summary = process_dataset(ds_id, segment_sec=args.segment)
        if summary:
            all_summaries.append(summary)
            for k, v in summary['label_dist'].items():
                total_dist[k] += v

    if len(ds_ids) > 1:
        total_segs = sum(total_dist.values())
        print()
        print("=" * 65)
        print(f"TOTAL: {total_segs} segmen dari {len(all_summaries)} video")
        print(f"  0=Aman      : {total_dist[0]:3d} ({total_dist[0]/total_segs*100:.1f}%)")
        print(f"  1=Hati-hati : {total_dist[1]:3d} ({total_dist[1]/total_segs*100:.1f}%)")
        print(f"  2=Waspada   : {total_dist[2]:3d} ({total_dist[2]/total_segs*100:.1f}%)")
        print(f"  3=Bahaya    : {total_dist[3]:3d} ({total_dist[3]/total_segs*100:.1f}%)")
        print()
        print(f"[SAVED] Output: {RESULTS_DIR}")

    # Simpan summary global
    if len(ds_ids) > 1:
        global_summary = {
            'total_segments': total_segs,
            'label_distribution': total_dist,
            'label_pct': {k: round(v/total_segs*100, 2) for k, v in total_dist.items()},
            'n_videos': len(all_summaries),
            'segment_sec': args.segment,
        }
        out_global = RESULTS_DIR / "gt_global_summary.json"
        with open(out_global, 'w') as f:
            json.dump(global_summary, f, indent=2)
        print(f"[SAVED] Global summary: {out_global}")


if __name__ == '__main__':
    main()
