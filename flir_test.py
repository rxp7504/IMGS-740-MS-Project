import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
cap.set(cv2.CAP_PROP_FPS, 9)

for _ in range(5):
    cap.grab()

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame")

print(f"Frame shape: {frame.shape}")  # let's see what we actually got

# Save raw
cv2.imwrite("thermal_raw.png", frame)

# Save colormap
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
cv2.imwrite("thermal_color.png", colormap)

print("Done! Saved thermal_raw.png and thermal_color.png")
