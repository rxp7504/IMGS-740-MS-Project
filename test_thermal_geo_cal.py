import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os

"""
Multi-image thermal geometric calibration script.
Loads all raw thermal tiffs, finds images with exactly 20 centroids,
runs calibration, and validates spacing.
"""

def find_centroids(thermal_tiff_path, threshold=100, min_area=5):
    """
    Load a raw thermal tiff and find handwarmer centroids.

    Args:
        thermal_tiff_path:  path to raw uint16 tiff
        threshold:          8-bit threshold for blob detection
        min_area:           minimum contour area to filter noise

    Returns:
        centroids:  list of (cx, cy) tuples, or None if not exactly 20 found
        thermal_8bit: normalized 8-bit image for visualization
    """
    thermal = cv2.imread(thermal_tiff_path, cv2.IMREAD_UNCHANGED)
    if thermal is None:
        return None, None

    thermal_8bit = cv2.normalize(
        thermal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    _, thresh = cv2.threshold(thermal_8bit, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))

    return centroids, thermal_8bit


def sort_grid(points, pts_w, pts_h):
    """
    Sort detected centroids into row-major grid order (left-right, top-bottom).
    """
    points = np.array(points)
    rows = sorted(points, key=lambda p: p[1])
    grid = []
    row_size = len(points) // pts_h
    for i in range(pts_h):
        row = rows[i * row_size:(i + 1) * row_size]
        row = sorted(row, key=lambda p: p[0])
        grid.extend(row)
    return np.array(grid, dtype=np.float32)


if __name__ == "__main__":

    print("=" * 70)
    print("MULTI-IMAGE THERMAL GEOMETRIC CALIBRATION")
    print("=" * 70)
    print()

    # --- Parameters ---
    cal_dir    = "static/geo_cal_captures"
    pts_w      = 5        # columns
    pts_h      = 4        # rows
    h_spacing  = 214.0    # mm (21.4 cm)
    v_spacing  = 246.7    # mm (24.67 cm)
    img_w, img_h = 160, 120

    # Lepton 3.5 known focal length from physical specs
    fov_h = 57.0
    fov_v = 44.0
    fx = (img_w / 2) / np.tan(np.radians(fov_h / 2))
    fy = (img_h / 2) / np.tan(np.radians(fov_v / 2))
    K_init = np.array([
        [fx,   0,  img_w / 2.0],
        [0,   fy,  img_h / 2.0],
        [0,    0,  1.0         ]
    ], dtype=np.float32)

    print(f"Initial focal lengths: fx={fx:.2f}, fy={fy:.2f}")

    # --- Build object points ---
    objp = np.zeros((pts_w * pts_h, 3), dtype=np.float32)
    for j in range(pts_h):
        for i in range(pts_w):
            objp[j * pts_w + i] = [i * h_spacing, j * v_spacing, 0]

    # --- Load all raw thermal tiffs ---
    tiff_paths = sorted(glob.glob(os.path.join(cal_dir, "thermal_*[!bit].tiff")))
    # Filter out any 8bit files just in case
    tiff_paths = [p for p in tiff_paths if "_8bit" not in p]
    print(f"Found {len(tiff_paths)} raw thermal tiff files")

    # --- Find valid images (exactly 20 centroids) ---
    all_obj_points = []
    all_img_points = []
    valid_paths    = []
    skipped        = 0

    for path in tiff_paths:
        centroids, thermal_8bit = find_centroids(path)

        if centroids is None:
            skipped += 1
            continue

        if len(centroids) != pts_w * pts_h:
            print(f"  SKIP {os.path.basename(path)} — {len(centroids)} centroids")
            skipped += 1
            continue

        sorted_pts = sort_grid(centroids, pts_w, pts_h)
        all_obj_points.append(objp.reshape(-1, 1, 3))
        all_img_points.append(sorted_pts.reshape(-1, 1, 2))
        valid_paths.append(path)

    print(f"Valid images: {len(valid_paths)} / {len(tiff_paths)}")
    print(f"Skipped:      {skipped}")
    print()

    if len(valid_paths) < 5:
        print("[ERROR] Need at least 5 valid images for reliable calibration.")
        exit()

    # --- Run calibration ---
    print("Running calibration...")
    flags = cv2.CALIB_USE_INTRINSIC_GUESS

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points,
        (img_w, img_h), K_init, None,
        flags=flags
    )

    print(f"RMS reprojection error: {ret:.4f}")
    print(f"Camera matrix K:\n{K}")
    print(f"Distortion coefficients: {dist}")

    # --- Save calibration ---
    np.save("_resources/thermal_K.npy",    K)
    np.save("_resources/thermal_dist.npy", dist)
    print("\nCalibration saved to _resources/")

    # --- Validate: check spacing on a sample image ---
    print("\nValidating spacing on first valid image...")
    sample_path = valid_paths[0]
    centroids, thermal_8bit = find_centroids(sample_path)
    sorted_pts = sort_grid(centroids, pts_w, pts_h)

    # Undistort the sample image
    thermal_raw = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
    thermal_undist = cv2.undistort(thermal_raw, K, dist)
    thermal_undist_8bit = cv2.normalize(
        thermal_undist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Find centroids in undistorted image
    centroids_undist, _ = find_centroids(sample_path)
    if centroids_undist and len(centroids_undist) == 20:
        # Undistort the centroid positions directly
        pts = np.array(sorted_pts).reshape(-1, 1, 2)
        pts_undist = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)

        # Measure horizontal and vertical spacings
        h_spacings = []
        v_spacings = []

        for row in range(pts_h):
            for col in range(pts_w - 1):
                idx = row * pts_w + col
                dx = pts_undist[idx + 1][0] - pts_undist[idx][0]
                dy = pts_undist[idx + 1][1] - pts_undist[idx][1]
                h_spacings.append(np.sqrt(dx**2 + dy**2))

        for col in range(pts_w):
            for row in range(pts_h - 1):
                idx = row * pts_w + col
                next_idx = (row + 1) * pts_w + col
                dx = pts_undist[next_idx][0] - pts_undist[idx][0]
                dy = pts_undist[next_idx][1] - pts_undist[idx][1]
                v_spacings.append(np.sqrt(dx**2 + dy**2))

        print(f"Horizontal spacing (pixels): mean={np.mean(h_spacings):.2f} std={np.std(h_spacings):.2f}")
        print(f"Vertical spacing   (pixels): mean={np.mean(v_spacings):.2f} std={np.std(v_spacings):.2f}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Calibration Results — RMS: {ret:.4f}", fontsize=14)

    # Show first 5 valid images with detected points
    for i, (ax, path) in enumerate(zip(axes.flat[:5], valid_paths[:5])):
        _, img_8bit = find_centroids(path)
        pts = all_img_points[i].reshape(-1, 2)
        display = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
        for j, (x, y) in enumerate(pts):
            cv2.circle(display, (int(x), int(y)), 2, (0, 255, 0), -1)
        ax.imshow(display)
        ax.set_title(os.path.basename(path)[-18:-5])
        ax.axis("off")

    # Show undistorted sample
    axes.flat[5].imshow(thermal_undist_8bit, cmap='inferno')
    axes.flat[5].set_title("Undistorted Sample")
    axes.flat[5].axis("off")

    plt.tight_layout()
    plt.savefig("_imgs/calibration_result.png")
    plt.show()

    print("\n")
    print("=" * 70)
    print("[SUCCESS] CALIBRATION COMPLETE")
    print("=" * 70)
