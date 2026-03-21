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

def prepare_pansharp(thermal_raw, rgb, warp_matrix_path,
                     shape_hires=(480, 640), shape_lores=(120, 160),
                     colormap=cv2.COLORMAP_INFERNO, verbose=False):
    """
    Prepare a Pseudo-MultiSpectral (PS-MS) image and PAN image for BDSD
    pansharpening from a raw thermal image and an RGB image.

    Pipeline:
        1. Crop thermal and RGB to the same aspect ratio
        2. Normalize thermal to 0-1 and create false color
        3. Resize RGB to match thermal hi-res resolution
        4. Load and apply registration matrix to align RGB to thermal
        5. Build 4-band PS-MS image (BGR false color + raw thermal gray)
        6. Downsample PS-MS to low-res then upsample back (PS-MS_HR')
        7. Create PAN image by grayscaling the registered RGB

    Args:
        thermal_raw:        2D uint16 numpy array, raw thermal image
        rgb:                3D float32 BGR numpy array (0-1), RGB image
        warp_matrix_path:   path to .npy file containing the 2x3 affine
                            warp matrix for RGB-to-thermal registration
        shape_hires:        tuple (H, W), target hi-res thermal resolution
                            (default: (480, 640))
        shape_lores:        tuple (H, W), target lo-res thermal resolution
                            (default: (120, 160))
        colormap:           OpenCV colormap for false color thermal
                            (default: cv2.COLORMAP_INFERNO)
        verbose:            Print shape info at each step (default: False)

    Returns:
        PS_MS_HR_p:     4-band float32 array (H, W, 4), upsampled PS-MS image
                        Bands: [B_fc, G_fc, R_fc, thermal_gray]
        pan:            float32 array (H, W), grayscale PAN image (0-1)
        raw_min:        float, min raw thermal value (for denormalization)
        raw_max:        float, max raw thermal value (for denormalization)
    """
    def crop_to_size(img, shape):
        diff_h = img.shape[0] - shape[0]
        diff_w = img.shape[1] - shape[1]
        start_h = diff_h // 2
        start_w = diff_w // 2
        return img[start_h:start_h + shape[0], start_w:start_w + shape[1]]

    # --- Crop thermal to hi-res shape and normalize ---
    thermal_cropped = crop_to_size(thermal_raw.astype(np.float32), shape_hires)
    raw_min = float(thermal_cropped.min())
    raw_max = float(thermal_cropped.max())
    thermal = cv2.normalize(thermal_cropped, None, 0.0, 1.0, cv2.NORM_MINMAX)

    if verbose:
        print(f"    Thermal cropped:   {thermal_cropped.shape} | min={raw_min:.1f} max={raw_max:.1f}")

    # --- Crop RGB to same aspect ratio then resize to hi-res ---
    target_h = int(rgb.shape[1] * (shape_hires[0] / shape_hires[1]))
    rgb_cropped = crop_to_size(rgb, [target_h, rgb.shape[1]])
    rgb_resized = cv2.resize(rgb_cropped, (shape_hires[1], shape_hires[0]))

    if verbose:
        print(f"    RGB resized:       {rgb_resized.shape}")

    # --- Load and apply registration matrix ---
    warp_matrix = np.load(warp_matrix_path)
    rgb_aligned = cv2.warpAffine(
        rgb_resized,
        warp_matrix,
        (shape_hires[1], shape_hires[0]),
        flags=cv2.INTER_LINEAR
    )

    if verbose:
        print(f"    RGB aligned:       {rgb_aligned.shape}")

    # --- Create false color thermal (BGR, uint8) ---
    thermal_color = cv2.applyColorMap(
        (thermal * 255).astype(np.uint8), colormap
    ).astype(np.float32) / 255.0

    # --- Build 4-band PS-MS_HR image ---
    PS_MS_HR = np.zeros((shape_hires[0], shape_hires[1], 4), dtype=np.float32)
    PS_MS_HR[:, :, 0:3] = thermal_color   # bands 0,1,2 = BGR false color
    PS_MS_HR[:, :, 3]   = thermal         # band 3 = raw thermal grayscale

    if verbose:
        print(f"    PS-MS-HR:          {PS_MS_HR.shape}")

    # --- Downsample then upsample to simulate incoming low-res image ---
    PS_MS_LR   = cv2.resize(PS_MS_HR,   (shape_lores[1], shape_lores[0]), interpolation=cv2.INTER_AREA)
    PS_MS_HR_p = cv2.resize(PS_MS_LR,   (shape_hires[1], shape_hires[0]), interpolation=cv2.INTER_NEAREST)

    if verbose:
        print(f"    PS-MS-LR:          {PS_MS_LR.shape}")
        print(f"    PS-MS-HR':         {PS_MS_HR_p.shape}")

    # --- Create PAN image from registered RGB ---
    pan = cv2.cvtColor(
        (rgb_aligned * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    if verbose:
        print(f"    PAN:               {pan.shape}")

    return PS_MS_HR_p, pan, raw_min, raw_max
