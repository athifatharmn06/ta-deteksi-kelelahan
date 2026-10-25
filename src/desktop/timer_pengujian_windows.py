import tkinter as tk
from tkinter import font as tkfont
import time
try:
    import winsound
except ImportError:  # Non-Windows fallback
    winsound = None

class DrowsinessTimerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PENGUJI DATA - TIMER INSTRUKSI")
        self.root.geometry("800x500")
        self.root.configure(bg='#FFFFFF')
        
        # Skenario Pengujian (Indonesian)
        self.instructions = [
            (0, 60, "MENYETIR NORMAL (Baseline)",
             "Posisi duduk nyaman, pandangan lurus ke depan.\n"
             "Kedua mata terbuka normal, jangan sengaja mengantuk.\n"
             "Minim gerakan kepala dan mulut, bernapas biasa.\n"
             "Tujuan: rekam kondisi normal untuk True Negative."),
            (60, 120, "UJI EAR: TUTUP MATA > 2 DETIK",
             "Tutup kedua mata selama 2-3 detik, lalu buka.\n"
             "Ulangi 4-6 kali dengan jeda 3-5 detik.\n"
             "Usahakan wajah tetap menghadap kamera.\n"
             "Tujuan: validasi True Positive untuk mata tertutup."),
            (120, 180, "UJI MAR: MENGUAP LEBAR",
             "Buka mulut selebar mungkin seperti menguap.\n"
             "Tahan 1-2 detik, lalu tutup. Ulangi 4-6 kali.\n"
             "Jaga kepala tetap stabil agar MAR terukur jelas.\n"
             "Tujuan: validasi deteksi mulut terbuka lebar."),
            (180, 240, "UJI POSE: MENUNDUK DALAM",
             "Tundukkan kepala ke bawah 30-45 derajat.\n"
             "Tahan 2-3 detik, kembali normal. Ulangi 4-6 kali.\n"
             "Jangan menutup wajah dengan tangan.\n"
             "Tujuan: validasi deteksi pitch kepala menunduk."),
            (240, 300, "GANGGUAN: BERBICARA, TERTAWA & BERNYANYI",
             "Lakukan bicara pelan, tertawa kecil, atau bernyanyi.\n"
             "Variasikan ekspresi mulut, tapi mata tetap terbuka.\n"
             "Jaga posisi kepala relatif stabil.\n"
             "Tujuan: uji False Positive saat mulut aktif."),
        ]
        
        self.state = "READY"
        self.start_time = 0
        self.elapsed_time = 0
        self.last_minute = -1
        
        # Fonts
        self.timer_font = tkfont.Font(family="Helvetica", size=80, weight="bold")
        self.instruction_font = tkfont.Font(family="Helvetica", size=30, weight="bold")
        self.desc_font = tkfont.Font(family="Helvetica", size=18)
        self.status_font = tkfont.Font(family="Helvetica", size=12)

        # UI Elements
        self.setup_ui()
        
        # Bind Key
        self.root.bind('<s>', self.start_timer)
        self.root.bind('<S>', self.start_timer)
        self.root.bind('<q>', lambda e: self.root.destroy())
        
        self.update_ui()

    def setup_ui(self):
        # Header
        self.header = tk.Label(self.root, text="SISTEM VALIDASI DETEKSI KELELAHAN", 
                       bg='#FFFFFF', fg='#3B3B3B', pady=20)
        self.header.pack()

        # Timer Label
        self.timer_label = tk.Label(self.root, text="00:00", font=self.timer_font, 
                        bg='#FFFFFF', fg='#111111')
        self.timer_label.pack(pady=10)

        # Instruction Box
        self.inst_container = tk.Frame(self.root, bg='#F5F5F5', padx=20, pady=20, 
                           highlightthickness=2, highlightbackground='#DDDDDD')
        self.inst_container.pack(fill='both', expand=True, padx=40, pady=20)

        self.inst_label = tk.Label(self.inst_container, text="TEKAN 'S' UNTUK MULAI", 
                       font=self.instruction_font, bg='#F5F5F5', fg='#0A7E73', wraplength=700)
        self.inst_label.pack()

        self.desc_label = tk.Label(self.inst_container, text="Pastikan kamera RPi sudah siap merekam.", 
                       font=self.desc_font, bg='#F5F5F5', fg='#555555', pady=10, wraplength=600)
        self.desc_label.pack()

        # Status Footer
        self.status_label = tk.Label(self.root, text="Status: Menunggu | Tekan 'q' untuk Keluar", 
                         bg='#FFFFFF', fg='#666666', pady=10)
        self.status_label.pack(side='bottom')

    def start_timer(self, event=None):
        if self.state == "READY":
            self.state = "RUNNING"
            self.start_time = time.time()
            self.last_minute = -1
            self.status_label.config(text="Status: PENGUJIAN SEDANG BERJALAN", fg='#0A7E73')
            self.inst_label.config(fg='#4A4A4A')

    def beep_minute(self):
        if winsound is None:
            return
        try:
            winsound.Beep(1000, 150)
            winsound.Beep(1000, 150)
        except Exception:
            pass

    def update_ui(self):
        if self.state == "RUNNING":
            self.elapsed_time = time.time() - self.start_time
            
            # Update Timer
            m = int(self.elapsed_time // 60)
            s = int(self.elapsed_time % 60)
            self.timer_label.config(text=f"{m:02d}:{s:02d}")

            if m != self.last_minute:
                self.last_minute = m
                self.beep_minute()
            
            # Update Instructions
            found = False
            for start, end, title, desc in self.instructions:
                if start <= self.elapsed_time < end:
                    self.inst_label.config(text=title)
                    self.desc_label.config(text=desc)
                    found = True
                    break
            
            if self.elapsed_time >= 300:
                self.state = "FINISHED"
                self.inst_label.config(text="PENGUJIAN SELESAI!", fg='#0A7E73')
                self.desc_label.config(text="Silakan cek file hasil pada Raspberry Pi.")
                self.status_label.config(text="Status: SELESAI", fg='#0A7E73')

        self.root.after(100, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessTimerApp(root)
    root.mainloop()
