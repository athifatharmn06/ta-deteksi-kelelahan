from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Laporan Progress" / "asset gambar"
FIGURES_DIR = ROOT / "Jurnal JNTETI" / "Figures"


def setup_ax(ax, title):
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)


def save_fig(fig, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_slide03_dataset_comparison():
    fig = plt.figure(figsize=(16, 9), facecolor="#f3f4f6")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 1.3], height_ratios=[1, 1], wspace=0.25, hspace=0.35)

    ax1 = fig.add_subplot(gs[:, 0])
    setup_ax(ax1, "DMD vs YawDD Coverage")
    categories = ["DMD Sessions", "DMD GT Yawn", "YawDD Total", "YawDD Talking", "YawDD Yawning"]
    values = [15, 38, 348, 100, 114]
    colors = ["#1d4ed8", "#2563eb", "#059669", "#10b981", "#34d399"]
    bars = ax1.barh(categories, values, color=colors)
    ax1.set_xlabel("Count", fontsize=12)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    for b in bars:
        ax1.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "DMD Evaluation Scope")
    pie_vals = [31, 7, 1]
    pie_labels = ["True Positive (31)", "False Negative (7)", "False Positive (1)"]
    pie_colors = ["#16a34a", "#ea580c", "#dc2626"]
    ax2.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", startangle=90, colors=pie_colors, textprops={"fontsize": 10})

    ax3 = fig.add_subplot(gs[1, 1])
    setup_ax(ax3, "YawDD Talking Stress Test")
    fp_rate = [32.0, 3.0, 0.0]
    cfg = ["C1", "C2", "C3"]
    c = ["#f59e0b", "#f97316", "#16a34a"]
    bars = ax3.bar(cfg, fp_rate, color=c, width=0.6)
    ax3.set_ylabel("False Positive Rate (%)")
    ax3.set_ylim(0, 35)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, fp_rate):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.1f}%", ha="center", fontsize=11)

    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis("off")
    ax4.set_facecolor("#e2e8f0")
    text = (
        "Dataset Summary\n\n"
        "DMD (IR Cabin, fatigue events)\n"
        "- 15 sessions\n"
        "- 38 visible-face yawning GT\n"
        "- Heterogeneous subjects\n\n"
        "YawDD (stress-test speech/noise)\n"
        "- 348 videos, 90 subjects\n"
        "- 100 talking videos for FP robustness\n"
        "- 114 yawning videos for generalization\n\n"
        "Goal: robust fatigue detection +\n"
        "near-zero false alarm in active speech"
    )
    ax4.text(0.03, 0.95, text, fontsize=14, va="top", ha="left", linespacing=1.4)

    fig.suptitle("Slide 3 - Perbandingan Dataset DMD dan YawDD", fontsize=26, fontweight="bold", y=0.99)
    save_fig(fig, "slide03_dataset_comparison.png")


def make_slide04_feature_landmarks_ear_mar_pitch():
    src = FIGURES_DIR / "qualitative_visual_results_subject6.png"
    if not src.exists():
        raise FileNotFoundError(f"Missing source image: {src}")

    img = mpimg.imread(src)
    h, w = img.shape[:2]

    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.2)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.set_title("Landmark-Based Visual Frames (Generated)", fontsize=18, fontweight="bold")
    ax_img.axis("off")

    rect_eye = patches.Rectangle((0.26 * w, 0.15 * h), 0.17 * w, 0.11 * h, linewidth=3, edgecolor="#2563eb", facecolor="none")
    rect_mouth = patches.Rectangle((0.34 * w, 0.45 * h), 0.20 * w, 0.16 * h, linewidth=3, edgecolor="#dc2626", facecolor="none")
    ax_img.add_patch(rect_eye)
    ax_img.add_patch(rect_mouth)
    ax_img.text(0.26 * w, 0.14 * h, "ROI Mata (EAR)", color="#1d4ed8", fontsize=12, fontweight="bold")
    ax_img.text(0.34 * w, 0.44 * h, "ROI Mulut (MAR)", color="#b91c1c", fontsize=12, fontweight="bold")

    ax_eq1 = fig.add_subplot(gs[0, 1])
    ax_eq1.set_facecolor("#e2e8f0")
    ax_eq1.axis("off")
    eq1 = (
        "Fitur Utama Ekstraksi\n\n"
        "Eye Aspect Ratio (EAR):\n"
        "EAR = (||p2-p6|| + ||p3-p5||) / (2 ||p1-p4||)\n\n"
        "Mouth Aspect Ratio (MAR):\n"
        "MAR = (A + B + C) / (2D)\n\n"
        "Head Pitch (Dot-Product Projection):\n"
        "Pitch ~ proj(forehead, nose, chin)"
    )
    ax_eq1.text(0.05, 0.9, eq1, fontsize=14, va="top", linespacing=1.5)

    ax_eq2 = fig.add_subplot(gs[1, 1])
    ax_eq2.set_facecolor("#dbeafe")
    ax_eq2.axis("off")
    eq2 = (
        "Interpretasi untuk Sistem\n\n"
        "1. EAR rendah durasi panjang -> indikasi microsleep\n"
        "2. MAR tinggi stabil -> kandidat yawning event\n"
        "3. Pitch menunduk persisten -> postural fatigue cue\n\n"
        "Ketiga sinyal dipakai bersama pada scoring fatigue"
    )
    ax_eq2.text(0.05, 0.9, eq2, fontsize=14, va="top", linespacing=1.6)

    fig.suptitle("Slide 4 - Landmark Mata, Mulut, dan Pitch Kepala", fontsize=26, fontweight="bold", y=0.99)
    save_fig(fig, "slide04_feature_landmarks_ear_mar_pitch.png")


def make_slide05_weight_system():
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)

    ax1 = fig.add_subplot(gs[:, 0])
    setup_ax(ax1, "Distribusi Bobot Fatigue Score")
    labels = ["Ocular (Eye)", "Labial (Mouth)", "Postural (Head)"]
    weights = [0.45, 0.25, 0.30]
    colors = ["#1d4ed8", "#dc2626", "#16a34a"]
    wedges, texts, autotexts = ax1.pie(weights, labels=labels, autopct="%1.0f%%", startangle=100, colors=colors, pctdistance=0.78)
    centre_circle = plt.Circle((0, 0), 0.48, fc="#f8fafc")
    ax1.add_artist(centre_circle)
    ax1.text(0, 0, "Proposed\nWeights", ha="center", va="center", fontsize=16, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Kontribusi Penalti per Event")
    events = ["Eye Closure", "Yawn Episode", "Head Nodding"]
    penalty = [45, 25, 30]
    bars = ax2.bar(events, penalty, color=colors)
    ax2.set_ylabel("Contribution to Score (%)")
    ax2.set_ylim(0, 55)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    for b, p in zip(bars, penalty):
        ax2.text(b.get_x() + b.get_width() / 2, p + 1.2, f"{p}%", ha="center", fontsize=11)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_facecolor("#e2e8f0")
    txt = (
        "Rasional Bobot\n\n"
        "- Ocular diberi bobot tertinggi (45%)\n"
        "  karena paling sensitif terhadap microsleep\n\n"
        "- Mouth diberi 25% agar tidak over-trigger\n"
        "  pada kondisi bicara/bernyanyi\n\n"
        "- Head diberi 30% sebagai sinyal postural\n"
        "  pelengkap untuk fatigue progression\n\n"
        "Output: satu Fatigue Score kumulatif"
    )
    ax3.text(0.04, 0.95, txt, va="top", fontsize=14, linespacing=1.5)

    fig.suptitle("Slide 5 - Weight System Multi-Signal", fontsize=26, fontweight="bold", y=0.99)
    save_fig(fig, "slide05_weight_system.png")


def make_slide06_yawn_state_machine():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#f8fafc")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    states = {
        "IDLE": (1.0, 6.5, "#dbeafe"),
        "BUILDING": (4.5, 6.5, "#fde68a"),
        "CONFIRMED": (8.0, 6.5, "#bbf7d0"),
        "COOLDOWN": (11.0, 6.5, "#fecaca"),
    }

    for name, (x, y, color) in states.items():
        box = patches.FancyBboxPatch((x, y), 2.3, 1.4, boxstyle="round,pad=0.2,rounding_size=0.12", fc=color, ec="#1f2937", lw=2)
        ax.add_patch(box)
        ax.text(x + 1.15, y + 0.7, name, ha="center", va="center", fontsize=14, fontweight="bold")

    def arrow(x1, y1, x2, y2, label):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2.2, color="#111827"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.3, label, fontsize=11, ha="center")

    arrow(3.3, 7.2, 4.5, 7.2, "MAR > threshold")
    arrow(6.8, 7.2, 8.0, 7.2, "Hold >= 1.5 s\n+ Peak Gate")
    arrow(10.3, 7.2, 11.0, 7.2, "MAR turun")
    arrow(12.1, 6.4, 2.1, 6.4, "Cooldown selesai")

    ax.add_patch(patches.FancyArrowPatch((6.0, 6.3), (6.0, 4.3), arrowstyle="->", mutation_scale=18, lw=2, color="#dc2626"))
    ax.text(6.0, 3.9, "Reset to IDLE jika\nOscillation Guard mendeteksi talking", color="#b91c1c", ha="center", fontsize=12)

    note = patches.FancyBboxPatch((0.8, 1.0), 12.6, 2.2, boxstyle="round,pad=0.25", fc="#e2e8f0", ec="#475569")
    ax.add_patch(note)
    ax.text(1.1, 2.95, "Penjelasan Komponen Validasi", fontsize=14, fontweight="bold", va="top")
    ax.text(
        1.1,
        2.55,
        "1) Adaptive Peak Gate: event harus mencapai bukaan mulut minimum fisiologis (MAR >= 0.50).\n"
        "2) Duration Gate: mouth opening stabil minimal 1.5 detik sebelum konfirmasi.\n"
        "3) Oscillation Guard: crossing MAR berulang cepat dianggap speech dan dibuang.",
        fontsize=12,
        va="top",
        linespacing=1.5,
    )

    fig.suptitle("Slide 6 - Yawn State Machine (Deterministik 4 Tahap)", fontsize=26, fontweight="bold", y=0.97)
    save_fig(fig, "slide06_yawn_state_machine.png")


def make_slide07_testing_procedure_pipeline():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#f8fafc")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    boxes = [
        (1.0, 8.0, 3.2, 2.0, "Input Video v6\n(DMD + YawDD)", "#dbeafe"),
        (5.0, 8.0, 3.5, 2.0, "Face ROI +\n478 Landmark", "#e0e7ff"),
        (9.2, 8.0, 3.6, 2.0, "Feature Extraction\nEAR, MAR, Pitch", "#ede9fe"),
        (13.5, 8.0, 4.0, 2.0, "Decision Engine\nWeight + State Machine", "#dcfce7"),
        (6.3, 4.5, 3.6, 2.0, "Event Matching\nvs Ground Truth", "#fef3c7"),
        (11.0, 4.5, 3.9, 2.0, "Confusion Matrix\nPrecision/Recall/F1", "#fee2e2"),
        (15.6, 4.5, 3.5, 2.0, "Profiling\nFPS, CPU, RAM", "#cffafe"),
    ]

    for x, y, w, h, txt, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.12", fc=color, ec="#1f2937", lw=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12, fontweight="bold")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=2.2, color="#111827"))

    arr(4.2, 9.0, 5.0, 9.0)
    arr(8.5, 9.0, 9.2, 9.0)
    arr(12.8, 9.0, 13.5, 9.0)
    arr(15.5, 8.0, 8.1, 6.5)
    arr(9.9, 5.5, 11.0, 5.5)
    arr(14.9, 5.5, 15.6, 5.5)

    ax.text(1.0, 2.7, "Fokus Uji v6", fontsize=15, fontweight="bold", color="#1e3a8a")
    ax.text(
        1.0,
        2.1,
        "- Validasi pemisahan fatigue asli vs perilaku natural pengemudi\n"
        "- Evaluasi robustnes Yawn State Machine dan Weight System\n"
        "- Profiling komputasi untuk target real-time edge deployment",
        fontsize=12.5,
        va="top",
        linespacing=1.5,
    )

    fig.suptitle("Slide 7 - Prosedur Pengujian dan Alur Evaluasi", fontsize=26, fontweight="bold", y=0.97)
    save_fig(fig, "slide07_testing_procedure_pipeline.png")


def make_slide08_detection_performance():
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.2, 1.2], height_ratios=[1, 1], wspace=0.25, hspace=0.3)

    ax_cm = fig.add_subplot(gs[:, 0])
    setup_ax(ax_cm, "Confusion Matrix (DMD, GT=38)")
    cm = np.array([[31, 1], [7, 0]])
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1], labels=["Pred Yawn", "Pred Non-Yawn"])
    ax_cm.set_yticks([0, 1], labels=["Actual Yawn", "Actual Non-Yawn*"])
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20, color="#111827", fontweight="bold")
    ax_cm.text(0.0, -0.25, "*Evaluasi utama berbasis event yawning visible-face", transform=ax_cm.transAxes, fontsize=9)

    ax_metrics = fig.add_subplot(gs[0, 1:])
    setup_ax(ax_metrics, "Metrik Utama Sistem")
    labels = ["Precision", "Recall", "F1-Score"]
    values = [96.9, 81.6, 88.6]
    colors = ["#16a34a", "#f59e0b", "#2563eb"]
    bars = ax_metrics.bar(labels, values, color=colors, width=0.55)
    ax_metrics.set_ylim(0, 100)
    ax_metrics.set_ylabel("Score (%)")
    ax_metrics.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, values):
        ax_metrics.text(b.get_x() + b.get_width() / 2, v + 1.1, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax_yawdd = fig.add_subplot(gs[1, 1])
    setup_ax(ax_yawdd, "YawDD Talking Robustness")
    c_labels = ["C1", "C2", "C3"]
    c_vals = [32.0, 3.0, 0.0]
    c_colors = ["#f59e0b", "#fb923c", "#16a34a"]
    bars = ax_yawdd.bar(c_labels, c_vals, color=c_colors)
    ax_yawdd.set_ylim(0, 35)
    ax_yawdd.set_ylabel("False Positive Rate (%)")
    ax_yawdd.grid(axis="y", linestyle="--", alpha=0.35)
    for b, v in zip(bars, c_vals):
        ax_yawdd.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=11)

    ax_note = fig.add_subplot(gs[1, 2])
    ax_note.axis("off")
    ax_note.set_facecolor("#e2e8f0")
    note = (
        "Analisis Kunci\n\n"
        "- Precision tinggi (96.9%) menunjukkan\n"
        "  alarm palsu sangat terkontrol\n\n"
        "- Recall 81.6% dipengaruhi kasus yawn\n"
        "  durasi pendek atau suppressed opening\n\n"
        "- Pada YawDD talking subset,\n"
        "  sistem full (C3) mencapai 0.0% FP"
    )
    ax_note.text(0.05, 0.95, note, va="top", fontsize=13.5, linespacing=1.5)

    fig.suptitle("Slide 8 - Hasil Pengujian Deteksi Algoritma", fontsize=26, fontweight="bold", y=0.99)
    save_fig(fig, "slide08_detection_performance.png")


def make_slide09_software_performance():
    fig = plt.figure(figsize=(16, 9), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    setup_ax(ax1, "Throughput & Latency")
    x = np.arange(2)
    proposed = [47.8, 20.9]
    ref = [12.4, 80.6]
    labels = ["FPS (higher better)", "Latency ms/frame\n(lower better)"]
    width = 0.35
    ax1.bar(x - width / 2, proposed, width, label="Proposed", color="#2563eb")
    ax1.bar(x + width / 2, ref, width, label="YOLOv8 Ref", color="#94a3b8")
    ax1.set_xticks(x, labels)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = fig.add_subplot(gs[0, 1])
    setup_ax(ax2, "Resource Consumption")
    x = np.arange(2)
    proposed2 = [39.9, 186]
    ref2 = [88.2, 640]
    labels2 = ["CPU (%)", "RAM (MB)"]
    ax2.bar(x - width / 2, proposed2, width, label="Proposed", color="#16a34a")
    ax2.bar(x + width / 2, ref2, width, label="YOLOv8 Ref", color="#f59e0b")
    ax2.set_xticks(x, labels2)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.set_facecolor("#e2e8f0")
    text = (
        "Kesimpulan Performa Software\n\n"
        "- Sistem mencapai 47.8 FPS pada CPU-only, memenuhi kebutuhan real-time untuk edge deployment kendaraan.\n"
        "- Rata-rata latensi 20.9 ms/frame, jauh di bawah referensi berbasis CNN berat.\n"
        "- Jejak sumber daya ringan (CPU 39.9%, RAM ~186 MB) memberikan headroom untuk modul ADAS lain.\n"
        "- Kombinasi akurasi deteksi + efisiensi komputasi mendukung kelayakan implementasi sistem pada perangkat non-GPU."
    )
    ax3.text(0.02, 0.88, text, fontsize=15, va="top", linespacing=1.6)

    fig.suptitle("Slide 9 - Performa Software dan Analisis Akhir", fontsize=26, fontweight="bold", y=0.97)
    save_fig(fig, "slide09_software_performance.png")


def main():
    make_slide03_dataset_comparison()
    make_slide04_feature_landmarks_ear_mar_pitch()
    make_slide05_weight_system()
    make_slide06_yawn_state_machine()
    make_slide07_testing_procedure_pipeline()
    make_slide08_detection_performance()
    make_slide09_software_performance()
    print("All assets generated successfully.")


if __name__ == "__main__":
    main()
