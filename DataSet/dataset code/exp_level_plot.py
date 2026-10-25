"""
exp_level_plot.py
Generate tabel dan plot dari hasil level_validation_results.csv

Usage:
    python exp_level_plot.py
"""

import csv, json
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib tidak tersedia. Hanya tabel teks yang digenerate.")

RESULTS_DIR = (Path(__file__).parent.parent.parent
               / "Jurnal JNTETI" / "Results" / "level_validation")
FIGURES_DIR = RESULTS_DIR.parent.parent / "Figures"


def load_results():
    path = RESULTS_DIR / "level_validation_results.csv"
    if not path.exists():
        print(f"[ERROR] File tidak ditemukan: {path}")
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(v):
    try: return float(v)
    except: return 0.0


# ── Tabel 1: Threshold Sensitivity ───────────────────────────────────────────
def print_threshold_table(rows):
    data = [r for r in rows
            if r['weight_config'] == 'W3_Proposed'
            and r['critical_override'].lower() in ('false', '0', '')]
    if not data:
        print("[WARN] Tidak ada data threshold sensitivity")
        return

    print("\n" + "="*70)
    print("TABEL THRESHOLD SENSITIVITY (W3=0.45/0.25/0.30, Tanpa Override)")
    print("="*70)
    header = f"{'Threshold':<16} {'Acc (%)':>8} {'Macro F1':>10} {'Rec Bahaya':>12} {'False Alarm':>13}"
    print(header)
    print("-"*len(header))
    for r in data:
        th = r['threshold_config']
        mark = " << Sistem saat ini" if 'T2' in th else ""
        print(f"{th:<16} {to_float(r['accuracy']):>7.1f}% "
              f"{to_float(r['macro_f1']):>9.1f}% "
              f"{to_float(r['recall_bahaya']):>11.1f}% "
              f"{to_float(r['false_alarm_rate']):>12.1f}%{mark}")


# ── Tabel 2: Weight Ablation ──────────────────────────────────────────────────
def print_weight_table(rows):
    data = [r for r in rows
            if r['threshold_config'] == 'T2_30_50_70'
            and r['critical_override'].lower() in ('false', '0', '')]
    if not data:
        print("[WARN] Tidak ada data weight ablation")
        return

    print("\n" + "="*78)
    print("TABEL ABLATION BOBOT (T2=30/50/70, Tanpa Override)")
    print("="*78)
    header = (f"{'Konfigurasi':<16} {'Eye':>5} {'Mouth':>6} {'Head':>5} "
              f"{'Acc':>7} {'MacroF1':>9} {'RecBahaya':>11} {'FalseAlarm':>12}")
    print(header)
    print("-"*len(header))
    for r in data:
        mark = " << Usulan" if r['weight_config'] == 'W3_Proposed' else ""
        print(f"{r['weight_config']:<16} {to_float(r['w_eye']):>5.2f} "
              f"{to_float(r['w_mouth']):>6.2f} {to_float(r['w_head']):>5.2f} "
              f"{to_float(r['accuracy']):>6.1f}% "
              f"{to_float(r['macro_f1']):>8.1f}% "
              f"{to_float(r['recall_bahaya']):>10.1f}% "
              f"{to_float(r['false_alarm_rate']):>11.1f}%{mark}")


# ── Tabel 3: Critical Override ────────────────────────────────────────────────
def print_override_table(rows):
    data = [r for r in rows
            if r['weight_config'] == 'W3_Proposed'
            and r['threshold_config'] == 'T2_30_50_70']
    if not data:
        return

    print("\n" + "="*65)
    print("PERBANDINGAN CRITICAL OVERRIDE (W3 + T2)")
    print("="*65)
    for r in data:
        mode = "B - Dengan Override" if r['critical_override'].lower() in ('true', '1') else "A - Tanpa Override "
        print(f"  {mode}: Acc={to_float(r['accuracy']):5.1f}% "
              f"F1={to_float(r['macro_f1']):5.1f}% "
              f"RecBahaya={to_float(r['recall_bahaya']):5.1f}% "
              f"FA={to_float(r['false_alarm_rate']):5.1f}%")


# ── Plot 1: Threshold Sensitivity Bar Chart ───────────────────────────────────
def plot_threshold_sensitivity(rows):
    if not HAS_MPL:
        return
    data = [r for r in rows
            if r['weight_config'] == 'W3_Proposed'
            and r['critical_override'].lower() in ('false', '0', '')]
    if not data:
        return

    labels    = [r['threshold_config'].replace('_', '\n') for r in data]
    macro_f1  = [to_float(r['macro_f1']) for r in data]
    rec_bahaya= [to_float(r['recall_bahaya']) for r in data]
    false_alarm=[to_float(r['false_alarm_rate']) for r in data]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, macro_f1,   width, label='Macro F1 (%)',        color='#2196F3')
    ax.bar(x,         rec_bahaya, width, label='Recall Bahaya (%)',    color='#F44336')
    ax.bar(x + width, false_alarm,width, label='False Alarm Rate (%)', color='#FF9800', alpha=0.8)

    # Tandai T2
    t2_idx = next((i for i, r in enumerate(data) if 'T2' in r['threshold_config']), None)
    if t2_idx is not None:
        ax.axvline(x=t2_idx, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(t2_idx + 0.05, ax.get_ylim()[1] * 0.95, 'Sistem\nini',
                fontsize=7, color='gray', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Nilai (%)')
    ax.set_title('Threshold Sensitivity Analysis (W3_Proposed, No Override)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "threshold_sensitivity_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ── Plot 2: Weight Ablation Bar Chart ─────────────────────────────────────────
def plot_weight_ablation(rows):
    if not HAS_MPL:
        return
    data = [r for r in rows
            if r['threshold_config'] == 'T2_30_50_70'
            and r['critical_override'].lower() in ('false', '0', '')]
    if not data:
        return

    labels    = [r['weight_config'].replace('_', '\n') for r in data]
    macro_f1  = [to_float(r['macro_f1']) for r in data]
    rec_bahaya= [to_float(r['recall_bahaya']) for r in data]
    false_alarm=[to_float(r['false_alarm_rate']) for r in data]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, macro_f1,   width, label='Macro F1 (%)',        color='#2196F3')
    ax.bar(x,         rec_bahaya, width, label='Recall Bahaya (%)',    color='#F44336')
    ax.bar(x + width, false_alarm,width, label='False Alarm Rate (%)', color='#FF9800', alpha=0.8)

    w3_idx = next((i for i, r in enumerate(data)
                   if r['weight_config'] == 'W3_Proposed'), None)
    if w3_idx is not None:
        ax.axvline(x=w3_idx, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
        ax.text(w3_idx + 0.05, ax.get_ylim()[1] * 0.95, 'Usulan',
                fontsize=7, color='gray', va='top')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Nilai (%)')
    ax.set_title('Ablation Study Bobot Kanal (T2=30/50/70, No Override)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "weight_ablation_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rows = load_results()
    if not rows:
        return

    print(f"[INFO] Loaded {len(rows)} result rows")

    print_threshold_table(rows)
    print_weight_table(rows)
    print_override_table(rows)

    plot_threshold_sensitivity(rows)
    plot_weight_ablation(rows)

    print("\n[DONE] Plot dan tabel selesai digenerate.")


if __name__ == '__main__':
    main()
