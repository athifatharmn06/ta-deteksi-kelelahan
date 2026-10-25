
try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"OpenCV Error: {e}")

try:
    print("Importing MediaPipe...")
    import mediapipe as mp
    print(f"MediaPipe: {mp.__version__}")
except Exception as e:
    print(f"MediaPipe Error: {e}")

try:
    print("Importing Pygame...")
    import pygame
    pygame.mixer.init()
    print("Pygame: initialized")
except Exception as e:
    print(f"Pygame Error: {e}")
