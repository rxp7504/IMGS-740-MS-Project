import cv2
import numpy as np
import matplotlib.pyplot as plt


def select_points(img, window_name, n_points=6):
    """
    Manually select corresponding points on an image by clicking.

    Args:
        img:          2D float32 grayscale image (0-1)
        window_name:  Title of the OpenCV window
        n_points:     Suggested number of points to select
    Returns:
        numpy array of (x, y) float32 points
    """
    points = []
    display = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    clone = display.copy()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, display)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                display[:] = clone.copy()
                for i, p in enumerate(points):
                    cv2.circle(display, p, 5, (0, 255, 0), -1)
                    cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(window_name, display)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click)
    cv2.imshow(window_name, display)

    print(f"    Click {n_points} points. Right click to undo. Press 'q' when done.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow(window_name)
    print(f"    {len(points)} points selected.")
    return np.float32(points)


if __name__ == "__main__":
    print("=" * 70)
    print("[TEST] CREATE REGISTRATION MATRIX - MANUAL")
    print("=" * 70)
    print("\n")

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

    # Display pre-registered images
    sbs = np.hstack((rgb, thermal))
    plt.imshow(cv2.cvtColor(sbs, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Pre-registered Images")
    plt.show()

    # Manual point selection
    print("\n    Select points on RGB image")
    pts_rgb = select_points(rgb, "RGB - Select Points")

    print("\n    Select the SAME points on Thermal image (same order!)")
    pts_thermal = select_points(thermal, "Thermal - Select Points")

    if len(pts_rgb) != len(pts_thermal):
        raise ValueError("Must select the same number of points on both images")
    if len(pts_rgb) < 3:
        raise ValueError("Need at least 3 point pairs for affine transform")

    # Estimate affine transform from point correspondences
    print("\n    Estimating affine transform...")
    warp_matrix, inliers = cv2.estimateAffine2D(pts_thermal, pts_rgb, method=cv2.RANSAC)
    n_inliers = int(inliers.sum()) if inliers is not None else len(pts_rgb)
    print(f"    Transform found! Inliers: {n_inliers}/{len(pts_rgb)}")

    # Apply to thermal
    aligned = cv2.warpAffine(
        thermal,
        warp_matrix,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR
    )

    # Display result
    sbs2 = np.hstack((rgb, aligned))
    plt.imshow(sbs2, cmap='gray')
    plt.axis("off")
    plt.title("After Registration: RGB (left) vs Aligned Thermal (right)")
    plt.show()
    
    # False color overlay: RGB in green channel, thermal in red channel
    overlay_color = np.zeros((*rgb.shape, 3), dtype=np.float32)
    overlay_color[:, :, 1] = rgb      # green = RGB
    overlay_color[:, :, 0] = aligned  # red = thermal
    
    plt.imshow(overlay_color)
    plt.axis("off")
    plt.title("Overlay: RGB (green) vs Thermal (red)")
    plt.show()
    
    # Save the transform
    np.save("_resources/warp_matrix.npy",warp_matrix)
    print("Transform saved to _resources folder")
    

    print("\n[SUCCESS] Image registration complete")
