import cv2
import sys
import time
import numpy as np

# Import the original drowsiness detection script
# Ensure it is in the same directory or in the python path
try:
    import v5_rpi5_drowsiness_detection as detect
except ImportError:
    print("Error: v5_rpi5_drowsiness_detection.py not found in current directory.")
    sys.exit(1)

# Path to the specific video file
VIDEO_PATH = r"C:\TA Deteksi Kelelahan\DataSet\DMD Dataset 1\gA_1_s5_2019-03-14T14;26;17+01;00_ir_face.mp4"

class VideoFileFeed(detect.RPiCameraFeed):
    """
    Mock RPiCameraFeed that reads from a video file instead of a camera.
    """
    def __init__(self, **kwargs):
        # Override dimensions to match video if possible, or use defaults
        self.width = kwargs.get('width', 640)
        self.height = kwargs.get('height', 480)
        self.use_rpicam = False
        
        print(f"Connecting to video source: {VIDEO_PATH}")
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        
        if not self.cap.isOpened():
            print(f"FAILED to open video: {VIDEO_PATH}")
            sys.exit(1)
            
        self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0 # Fallback
        # Keep references to REAL time functions for throttling
        self._real_time = time.time
        self._real_sleep = time.sleep
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30.0 # Fallback
        self.frame_delay = 1.0 / self.fps
        
        # Virtual Time Sync: Make the algorithm believe it is running at perfect video FPS
        self.virtual_time = self._real_time()
        self.last_real_frame_time = self._real_time()
        
        # Monkey-patch time.time IN the detect module
        def get_virtual_time():
            return self.virtual_time
        detect.time.time = get_virtual_time
        
        print(f"Video Resolution: {self.capture_width}x{self.capture_height} | FPS: {self.fps:.2f}")

    def read(self):
        # Throttling using REAL time functions
        now = self._real_time()
        elapsed = now - self.last_real_frame_time
        sleep_time = self.frame_delay - elapsed
        if sleep_time > 0:
            self._real_sleep(sleep_time)
            
        ret, frame = self.cap.read()
        self.last_real_frame_time = self._real_time()
        
        if not ret:
            print("End of video file reached.")
            return False, None
        
        # Increment virtual time by exactly one frame duration
        self.virtual_time += self.frame_delay
        
        # Use the original letterboxing logic
        return True, self._letterbox(frame)

    def isOpened(self):
        return self.cap and self.cap.isOpened()

    def release(self):
        self.cap.release()

# Monkey-patch the RPiCameraFeed class in the original module
detect.RPiCameraFeed = VideoFileFeed

if __name__ == "__main__":
    print("--- Running Drowsiness Detection on Video ---")
    print("Note: Calibration will still require ~3 seconds of 'steady' video footage.")
    try:
        detect.main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
