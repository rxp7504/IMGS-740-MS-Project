import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load images
    rgb = cv2.imread("_imgs/rgb_test.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.imread("_imgs/thermal_img.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.rotate(thermal, cv2.ROTATE_180)  # thermal camera is upside down

    # Normalize
    cv2.normalize(rgb, rgb, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(thermal, thermal, 0, 1, cv2.NORM_MINMAX)

    # Resize RGB to be 4x the thermal image
    rgb = cv2.resize(rgb, (thermal.shape[1] * 4, thermal.shape[0] * 4))
    print(f"    RGB image resized to {rgb.shape}")

    # Resize thermal to match the down-res RGB
    thermal = cv2.resize(thermal, (thermal.shape[1] * 4, thermal.shape[0] * 4))
    print(f"    Thermal image resized to {thermal.shape}")
