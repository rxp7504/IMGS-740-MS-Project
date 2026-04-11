import cv2
import numpy as np
import matplotlib.pyplot as plt
from thermal_camera import ThermalCamera

# Capture a burst of frames
with ThermalCamera() as cam:
    cam.start(mode="standard")
    
    frames = []
    for i in range(16):  # 16 frames like FLIR Ultramax
        frame = cam.capture()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        print(f"Captured frame {i+1}/16")

frames = np.array(frames)
print(f"Burst shape: {frames.shape}")

# Estimate shifts between frames using phase correlation
reference = frames[0]
shifts = []
for i, frame in enumerate(frames[1:]):
    shift, response = cv2.phaseCorrelate(reference, frame)
    shifts.append(shift)
    print(f"Frame {i+1}: shift = ({shift[0]:.3f}, {shift[1]:.3f}) px")

# Plot shift distribution
shifts = np.array(shifts)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(shifts[:, 0], shifts[:, 1])
plt.xlabel("X shift (px)")
plt.ylabel("Y shift (px)")
plt.title("Sub-pixel shift distribution")
plt.axhline(0, color='r', linewidth=0.5)
plt.axvline(0, color='r', linewidth=0.5)
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(shifts[:, 0], label='X')
plt.plot(shifts[:, 1], label='Y')
plt.xlabel("Frame")
plt.ylabel("Shift (px)")
plt.title("Shift over time")
plt.legend()
plt.tight_layout()
plt.savefig("_imgs/shift_analysis.png")
plt.show()
