import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Single-image geometric correction using grid + homography
Stable version (no TPS issues)
"""

def sort_grid(points, pts_w, pts_h):
    """
    Sort detected centroids into row-major grid order
    """
    points = np.array(points)

    # sort by y (rows)
    rows = sorted(points, key=lambda p: p[1])

    grid = []
    row_height = len(points) // pts_h

    for i in range(pts_h):
        row = rows[i*row_height:(i+1)*row_height]
        row = sorted(row, key=lambda p: p[0])  # sort by x
        grid.extend(row)

    return np.array(grid, dtype=np.float32)


if __name__ == "__main__":
    
    print("="*70)
    print("THERMAL DISTORTION CORRECTION (HOMOGRAPHY)")
    print("="*70)
    
    # Load image
    thermal = cv2.imread("_cal_imgs/thermal_cal_2.tiff", cv2.IMREAD_UNCHANGED)
    thermal_8bit = cv2.normalize(thermal, None, 0, 255,
                                 cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    h_img, w_img = thermal_8bit.shape

    # Grid dimensions
    pts_w = 5
    pts_h = 4

    # ----------------------------------------
    # Detect hand warmers
    # ----------------------------------------
    _, thresh = cv2.threshold(thermal_8bit, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for c in contours:
        if cv2.contourArea(c) < 50:  # filter noise
            continue

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))

    print(f"Detected {len(centroids)} blobs")

    if len(centroids) != pts_w * pts_h:
        print("[ERROR] Wrong number of detected points")
        exit()

    # ----------------------------------------
    # Sort into grid
    # ----------------------------------------
    src_pts = sort_grid(centroids, pts_w, pts_h)

    # ----------------------------------------
    # Create ideal grid (pixel space)
    # ----------------------------------------
    margin = 10
    grid_w = w_img - 2 * margin
    grid_h = h_img - 2 * margin

    ideal_pts = []
    for j in range(pts_h):
        for i in range(pts_w):
            x = margin + i * (grid_w / (pts_w - 1))
            y = margin + j * (grid_h / (pts_h - 1))
            ideal_pts.append([x, y])

    ideal_pts = np.array(ideal_pts, dtype=np.float32)
    
    print(f"Src Pts: {src_pts}")
    print(f"Ideal Pts: {ideal_pts}")

    # ----------------------------------------
    # Compute homography
    # ----------------------------------------
    H, _ = cv2.findHomography(src_pts, ideal_pts, cv2.RANSAC)

    corrected = cv2.warpPerspective(
        thermal_8bit,
        H,
        (w_img, h_img)
    )

    # ----------------------------------------
    # Visualization
    # ----------------------------------------
    display = cv2.cvtColor(thermal_8bit, cv2.COLOR_GRAY2BGR)

    for i, (x, y) in enumerate(src_pts):
        cv2.circle(display, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(display, str(i), (int(x)+4, int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    # Draw ideal grid on corrected image
    corrected_display = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    for (x, y) in ideal_pts:
        cv2.circle(corrected_display, (int(x), int(y)), 3, (0, 0, 255), -1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.title("Original (Detected Points)")
    plt.imshow(display)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Corrected (Ideal Grid)")
    plt.imshow(corrected_display)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    np.save("_resources/thermal_H.npy", H)
    print("Homography saved to _resources/thermal_H.npy")

    print("\n")
    print("="*70)
    print("[SUCCESS] CORRECTION COMPLETE")
    print("="*70)
