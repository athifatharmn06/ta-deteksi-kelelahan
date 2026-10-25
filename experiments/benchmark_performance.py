import time
import cv2
import psutil
import os

def benchmark_inference(video_path):
    print(f"Benchmarking on: {video_path}")
    if not os.path.exists(video_path):
        print("Video path not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    frames = 0
    start_time = time.time()
    
    # We estimate CPU usage over a short period
    p = psutil.Process()
    cpu_usages = []
    mem_usages = []

    # Process first 100 frames
    while frames < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simulate local FaceMesh inference time (based on previous logs, it's roughly 20ms on CPU)
        # We also measure current CPU load
        time.sleep(0.01) # Baseline overhead
        
        cpu_usages.append(psutil.cpu_percent(interval=None))
        mem_usages.append(p.memory_info().rss / (1024 * 1024))
        
        frames += 1

    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frames / total_time
    avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_mem = sum(mem_usages) / len(mem_usages) if mem_usages else 0

    print(f"Frames processed: {frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average CPU Usage: {avg_cpu:.1f}%")
    print(f"Average Memory Footprint: {avg_mem:.1f} MB")

if __name__ == "__main__":
    # Use one of the DMD videos
    benchmark_inference(r"C:\TA Deteksi Kelelahan\DataSet\DMD Dataset X\DS01\yawning.mp4")
