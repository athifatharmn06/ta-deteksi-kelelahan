# Raspberry Pi Setup Notes

Dokumen ini untuk membuat ulang environment di Raspberry Pi OS dan menjalankan runtime utama dari struktur repo terbaru.

## 1. Prasyarat sistem

Disarankan:

- Python 3.10 atau 3.11.
- Kamera Raspberry Pi yang bisa diakses lewat `rpicam-vid`.
- Paket build dasar jika ada dependency yang perlu compile:
  - `python3-dev`
  - `build-essential`
  - `cmake`
- Tkinter hanya diperlukan untuk script GUI tertentu:
  - `python3-tk`

## 2. Buat virtual environment

```bash
cd ta-deteksi-kelelahan
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

`myenv/` tidak boleh di-commit.

## 3. Verifikasi dependency

```bash
python -c "import cv2, mediapipe, numpy, pandas, psutil, pygame, gpiozero; print('ok')"
```

Jika `tensorflow` gagal di Raspberry Pi, cek apakah eksperimen yang dijalankan memang membutuhkan TensorFlow. Sebagian besar runtime RPi terbaru tidak memakai TensorFlow.

## 4. Jalankan runtime utama

```bash
python src/rpi/v8_rpi5_drowsiness_detection.py
```

Kontrol utama:

- `c`: mulai kalibrasi setelah jeda 1.5 detik.
- GPIO 26: fungsi sama seperti tombol `c`.
- `r`: reset kalibrasi.
- `s`: screenshot.
- `q`: keluar.

Audio runtime berada di `src/rpi/audio/`. Jangan pindahkan folder ini tanpa menyesuaikan `AUDIO_DIR` di `src/rpi/v8_rpi5_drowsiness_detection.py`.

## 5. Catatan paket yang sering berbeda di Raspberry Pi

- `opencv-python` dan `opencv-contrib-python` kadang bermasalah tergantung wheel dan versi OS. Jika install pip gagal, pertimbangkan `sudo apt install python3-opencv` lalu sesuaikan environment.
- `rpicam-vid` bukan package pip. Pastikan command tersebut tersedia di Raspberry Pi OS.
- `gpiozero` dipakai untuk tombol GPIO 26. Jika backend GPIO bermasalah, cek paket OS seperti `python3-gpiozero` atau backend GPIO yang sesuai board.
- `winsound` hanya tersedia di Windows dan tidak perlu di-install di Raspberry Pi.
- `tkinter` biasanya berasal dari package OS, bukan pip.
