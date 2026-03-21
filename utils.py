import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_registration_matrix(img_fixed, img_moving, filename):
    """
    Manually select corresponding points on two images and estimate
    an affine transform matrix, then save it to disk.

    Args:
        img_fixed:  3-channel float32 BGR image (0-1), the reference image
        img_moving: 3-channel float32 BGR image (0-1), the image to align
        filename:   path to save the warp matrix (.npy)
    Returns:
        warp_matrix: 2x3 float32 affine transform matrix
    """
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

    # Convert to grayscale for point selection
    gray_fixed  = cv2.cvtColor((img_fixed  * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_moving = cv2.cvtColor((img_moving * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    print("\n    Select points on fixed image")
    pts_fixed = select_points(gray_fixed, "Fixed Image - Select Points")

    print("\n    Select the SAME points on moving image (same order!)")
    pts_moving = select_points(gray_moving, "Moving Image - Select Points")

    if len(pts_fixed) != len(pts_moving):
        raise ValueError("Must select the same number of points on both images")
    if len(pts_fixed) < 3:
        raise ValueError("Need at least 3 point pairs for affine transform")

    # Estimate affine transform
    print("\n    Estimating affine transform...")
    warp_matrix, inliers = cv2.estimateAffine2D(pts_moving, pts_fixed, method=cv2.RANSAC)
    n_inliers = int(inliers.sum()) if inliers is not None else len(pts_fixed)
    print(f"    Transform found! Inliers: {n_inliers}/{len(pts_fixed)}")

    # Save and return
    np.save(filename, warp_matrix)
    print(f"    Warp matrix saved to {filename}")
    return warp_matrix
    
    
def overlay_imgs(rgb,thermal):
    rgb_gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    thermal_gray = cv2.cvtColor((thermal     * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    overlay_color = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
    overlay_color[:, :, 1] = rgb_gray      # green = RGB
    overlay_color[:, :, 0] = thermal_gray  # red = thermal

    plt.imshow(overlay_color)
    plt.axis("off")
    plt.title("Overlay: RGB (green) vs Thermal (red)")
    plt.show()

def contrast_enhance(img):
    img_8bit = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_8bit).astype(np.float32) / 255.0
    return img_enhanced
