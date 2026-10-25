# Sistem Deteksi Kelelahan Pengemudi Real-Time Berbasis Raspberry Pi 5 dan MediaPipe Face Mesh

Repository ini berisi kode sumber dan eksperimen Tugas Akhir (TA) mengenai pengembangan sistem deteksi dini kelelahan pengemudi kendaraan secara *real-time* menggunakan *computer vision* pada *single-board computer* Raspberry Pi 5.

Sistem memanfaatkan kamera inframerah (IR) / Raspberry Pi Camera Module dengan framework MediaPipe Face Mesh dan OpenCV untuk mengekstraksi landmark wajah pengemudi. Pendekatan deteksi menggabungkan tiga indikator visual utama secara *rule-based* dengan akumulasi temporal multi-level untuk mengidentifikasi tingkat kelelahan secara akurat serta meminimalkan *false alarm*.

---

## Fitur Utama

- **Deteksi Mata (EAR & PERCLOS)**  
  Menghitung *Eye Aspect Ratio* (EAR) untuk mendeteksi kedipan mata (*blink*), kedipan berdurasi panjang (*prolonged blink*), dan *microsleep*. Sistem juga menghitung nilai PERCLOS (*Percentage of Eye Closure*) dalam jendela waktu bergerak untuk mengukur penurunan kewaspadaan pengemudi.

- **Deteksi Mulut (MAR & Talking Suppression)**  
  Menghitung *Mouth Aspect Ratio* (MAR) untuk mendeteksi frekuensi dan durasi menguap (*yawning*). Dilengkapi dengan mekanisme *oscillation guard* (analisis frekuensi osilasi bukaan mulut) untuk membedakan menguap dari aktivitas berbicara sehingga menekan *false positive*.

- **Estimasi Pose Kepala (Head Pose Pitch)**  
  Mendeteksi kemiringan kepala (*head nod / head drop*) saat pengemudi mulai kehilangan kesadaran atau tertidur.

- **Akumulator Temporal Multi-Level**  
  Menggabungkan skor dari mata, mulut, dan pose kepala dengan bobot dinamis dan mekanisme *decay* waktu. Status kelelahan dibagi menjadi tiga tingkat peringatan:
  - **Tingkat 1 - Hati-hati**: Terdeteksi indikasi awal kantuk (kedipan melambat atau menguap ringan).
  - **Tingkat 2 - Waspada**: Terdeteksi menguap berulang atau penurunan bukaan mata secara konsisten.
  - **Tingkat 3 - Bahaya**: Terdeteksi *microsleep* atau kepala tertunduk yang memerlukan intervensi segera.

- **Audio Manager & Voice Warning**  
  Menyediakan peringatan suara interaktif (*voice prompt*) berbahasa Indonesia dan alarm audio peringatan bertingkat sesuai level kelelahan.

- **Dukungan Tombol Kalibrasi Fisik (GPIO 26)**  
  Mendukung push button eksternal yang terhubung ke pin GPIO 26 Raspberry Pi untuk memicu kalibrasi baseline wajah pengemudi secara praktis sebelum berkendara.

---

## Struktur Direktori

```text
ta-deteksi-kelelahan/
├── src/
│   ├── rpi/                 # Implementasi runtime untuk Raspberry Pi 5
│   │   ├── audio/           # File audio alarm dan instruksi suara (WAV/MP3)
│   │   ├── v8_rpi5_drowsiness_detection.py  # Entry point utama runtime RPi5
│   │   ├── v6_rpi5_drowsiness_detection.py  # Versi stabil runtime RPi5
│   │   ├── test_lux_rpi5.py # Pengujian ekstraksi landmark pada kondisi lux
│   │   └── ...
│   └── desktop/             # Implementasi runtime untuk pengujian desktop/webcam
│       ├── demo.py          # Demo pengujian pada file video lokal atau webcam
│       ├── monitoring_subjek2.py
│       └── timer_pengujian_windows.py
├── DataSet/                 # Benchmark dataset, parser, dan script evaluasi
│   ├── dataset code/        # Script evaluasi model, ablation study, dan sweep
│   ├── DMD Dataset 1-15/    # Ground truth anotasi CSV dan JSON per subjek
│   └── YawDD/               # Anotasi dan ground truth dataset YawDD
├── experiments/             # Script eksperimen performa, latensi, dan ablasi
├── visualizations/          # Script pembuatan grafik, Pareto plot, dan visualisasi
├── scripts/
│   └── analysis/            # Script analisis mendalam untuk yawning & head drop
├── docs/                    # Dokumentasi teknis setup Raspberry Pi
│   └── raspberrypi_setup.md
├── requirements.txt         # Daftar pustaka / dependensi Python
└── README.md
```

---

## Kebutuhan Sistem

### Perangkat Keras
- Raspberry Pi 5 (RAM 4GB atau 8GB direkomendasikan).
- Raspberry Pi Camera Module (misal Camera Module V2 IMX219 atau kamera NoIR IR).
- LED IR Illuminator (untuk pengujian kondisi minim cahaya / malam hari).
- Speaker atau buzzer audio 3.5mm / USB.
- Push button untuk kalibrasi (terhubung ke GPIO 26 dan GND).

### Perangkat Lunak
- Raspberry Pi OS (64-bit, Debian Bookworm).
- Python 3.10 atau 3.11.
- `rpicam-apps` (`rpicam-vid`) untuk stream video perangkat keras pada Raspberry Pi OS.

---

## Instalasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/athifatharmn06/ta-deteksi-kelelahan.git
   cd ta-deteksi-kelelahan
   ```

2. **Buat dan aktifkan virtual environment:**
   - **Linux / Raspberry Pi OS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependensi:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

> Panduan konfigurasi khusus untuk sistem operasi Raspberry Pi OS tersedia di file [docs/raspberrypi_setup.md](docs/raspberrypi_setup.md).

---

## Cara Menjalankan

### 1. Menjalankan pada Raspberry Pi 5 (Utama)
Pastikan kamera terpasang dan dapat diakses dengan baik, kemudian jalankan:

```bash
python src/rpi/v8_rpi5_drowsiness_detection.py
```

**Kontrol Operasi:**
- **Tombol GPIO 26 / Tekan tombol `c`**: Memulai proses kalibrasi (posisi normal mata, mulut, dan pose kepala) dengan jeda 1.5 detik.
- **Tekan `r`**: Reset kalibrasi dan kembalikan sistem ke mode inisialisasi.
- **Tekan `h`**: Menampilkan atau menyembunyikan grafik metrik (*EAR/MAR/Pitch/Score*).
- **Tekan `u`**: Beralih antara tampilan UI lengkap dan mode ringkas (*compact*).
- **Tekan `s`**: Menyimpan tangkapan layar (*screenshot*) frame saat ini.
- **Tekan `q`**: Keluar dari aplikasi.

### 2. Menjalankan Demo pada Desktop / Laptop
Untuk pengujian atau demonstrasi menggunakan webcam laptop atau file video rekaman:

```bash
python src/desktop/demo.py
```

---

## Dataset & Evaluasi

Sistem ini dievaluasi secara kuantitatif menggunakan dataset publik pengemudi kendaraan:
1. **Driver Monitoring Dataset (DMD)**: Sesi `s5` (kamera IR real car) yang mencakup 15 subjek dalam berbagai kondisi pencahayaan dan variasi penggunaan kacamata.
2. **YawDD (Yawning Detection Dataset)**: Dataset benchmark untuk evaluasi fitur menguap dan kemampuan membedakan menguap dari berbicara.

Skrip evaluasi model terhadap dataset, benchmarking FPS/latensi, dan studi sensitivitas parameter tersedia pada folder `DataSet/dataset code/` dan `experiments/`.

---

## Sitasi & Referensi

Jika menggunakan materi atau kode dari repository ini dalam penelitian, harap menyertakan referensi ke:
- **DMD (Driver Monitoring Dataset)**: Vicomtech Driver Monitoring Dataset ([https://dmd.vicomtech.org/](https://dmd.vicomtech.org/)).
- **MediaPipe Face Mesh**: Google MediaPipe Solution ([https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)).
