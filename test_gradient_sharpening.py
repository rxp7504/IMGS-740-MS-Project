"""
Thermally-Guided Gradient Domain Pansharpening
===============================================

Concept:
    Sharpen a low-resolution thermal image using a high-resolution RGB image,
    but only inject RGB spatial detail where the thermal image confirms real
    thermal contrast exists. This prevents hallucinating RGB-only features
    (e.g. shirt prints) into the thermal image.

Method:
    1. Compute gradient magnitude maps from both thermal and RGB images
    2. Build a gradient agreement map (high where both modalities have edges)
       using harmonic mean — penalizes imbalance, requires both to be high
    3. Use agreement map as enhanced gradient magnitude
    4. Reconstruct image from enhanced gradient field via Poisson solver
    5. Apply linear scale correction to preserve radiometric calibration
"""

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import median_filter, gaussian_filter
import tifffile
import os


# =============================================================================
#  Poisson Reconstruction
# =============================================================================

def poisson_reconstruct(gx: np.ndarray, gy: np.ndarray,
                        mean_val: float) -> np.ndarray:
    """
    Reconstruct an image from its gradient fields using a DCT-based
    Poisson solver with Neumann boundary conditions.

    Args:
        gx:       horizontal gradient field (H x W)
        gy:       vertical gradient field (H x W)
        mean_val: DC offset — mean of original image

    Returns:
        I: reconstructed image (H x W)
    """
    h, w = gx.shape

    # Divergence of gradient field
    div = np.zeros((h, w), dtype=np.float64)
    div[:, :-1] += np.diff(gx, axis=1)
    div[:-1, :] += np.diff(gy, axis=0)

    # Solve via DCT-2 (Neumann boundary conditions)
    div_dct = dctn(div, type=2, norm='ortho')

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    eigenvalues = (2*np.cos(np.pi*x/w) - 2) + \
                  (2*np.cos(np.pi*y/h) - 2)
    eigenvalues[0, 0] = 1  # avoid division by zero

    solution = div_dct / eigenvalues
    solution[0, 0] = 0  # zero DC — corrected separately

    I = idctn(solution, type=2, norm='ortho')
    I = I - I.mean() + mean_val

    return I.astype(np.float32)


# =============================================================================
#  Gradient Sharpening
# =============================================================================
def gradient_sharpen(thermal: np.ndarray, pan: np.ndarray,
                     sigma_vis: float = 1.0):

    # Work in float32 throughout to halve memory usage
    thermal64 = thermal.astype(np.float32)
    pan_filtered = gaussian_filter(
        median_filter(pan.astype(np.float32), size=3), sigma=sigma_vis)

    # Gradient magnitudes
    def grad_magnitude(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        del gx, gy
        return mag

    gradMag_therm = grad_magnitude(thermal64)
    gradMag_vis   = grad_magnitude(pan_filtered)
    del pan_filtered

    gmt_max       = gradMag_therm.max() + 1e-8
    gradMag_therm /= gmt_max
    gradMag_vis   /= gmt_max

    # Harmonic mean agreement map
    agreement     = 2 * (gradMag_therm * gradMag_vis) / \
                    (gradMag_therm + gradMag_vis + 1e-8)
    del gradMag_vis
    agreement    /= (agreement.max() + 1e-8)
    gradMag_blend = agreement
    del gradMag_therm

    # Signed gradients for direction
    gy_t, gx_t = np.gradient(thermal64)
    Gmag_base   = np.sqrt(gx_t**2 + gy_t**2)

    gx_unit = (gx_t / (Gmag_base + 1e-8)).astype(np.float32)
    gy_unit = (gy_t / (Gmag_base + 1e-8)).astype(np.float32)
    del gx_t, gy_t, Gmag_base

    gx_enhanced = (gradMag_blend * gx_unit).astype(np.float64)
    gy_enhanced = (gradMag_blend * gy_unit).astype(np.float64)
    del gx_unit, gy_unit, gradMag_blend

    # Reconstruct
    I_sharpened = poisson_reconstruct(gx_enhanced, gy_enhanced,
                                      float(thermal.mean()))
    del gx_enhanced, gy_enhanced

    # Linear scale correction
    A = np.vstack([I_sharpened.ravel(), np.ones(I_sharpened.size)]).T
    p, _, _, _ = np.linalg.lstsq(A, thermal.ravel(), rcond=None)
    I_sharpened = (I_sharpened * p[0] + p[1]).astype(np.float32)

    return I_sharpened, agreement, p

# =============================================================================
#  Radiometric Conversion
# =============================================================================

def normalized_to_celsius(img: np.ndarray,
                           raw_min: float, raw_max: float) -> np.ndarray:
    """
    Convert normalized [0-1] thermal image back to Celsius.
    Lepton radiometry: raw value = Kelvin * 100
    """
    raw = img * (raw_max - raw_min) + raw_min
    return (raw / 100.0) - 273.15


# =============================================================================
#  Save image utilities
# =============================================================================

def save_gray(path: str, img: np.ndarray) -> None:
    """Save a float image as a normalized 8-bit grayscale PNG."""
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, out)


def save_heatmap(path: str, img: np.ndarray) -> None:
    """Save a float image as a hot colormap PNG."""
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(path, cv2.applyColorMap(out, cv2.COLORMAP_HOT))


def save_diverging(path: str, img: np.ndarray) -> None:
    """Save a signed float image as a cool colormap PNG."""
    vmax = np.abs(img).max() + 1e-8
    out  = ((img / vmax + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.applyColorMap(out, cv2.COLORMAP_COOL))


# =============================================================================
#  Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[TEST] THERMALLY-GUIDED GRADIENT DOMAIN PANSHARPENING")
    print("=" * 70)

    print(f"Available memory: {os.popen('free -h').read()}")


    # -------------------------------------------------------------------------
    #  Load Images
    # -------------------------------------------------------------------------
    pan_raw = cv2.imread("_imgs/pan.tiff", cv2.IMREAD_UNCHANGED)
    pan     = pan_raw.astype(np.float64)

    PS_MS   = tifffile.imread("_imgs/PS_MS_HR_p.tiff")
    thermal = PS_MS[:, :, -1].astype(np.float32) if PS_MS.ndim == 3 \
              else PS_MS.astype(np.float32)

    # Radiometric calibration constants from capture script
    THERMAL_MIN_RAW = 29242.0
    THERMAL_MAX_RAW = 30829.0

    print(f"Thermal shape: {thermal.shape}")
    print(f"PAN shape:     {pan.shape}")
    print(f"Thermal range: {thermal.min():.4f} to {thermal.max():.4f}")

    # -------------------------------------------------------------------------
    #  Run Sharpening
    # -------------------------------------------------------------------------
    print("\nRunning gradient sharpening...")
    I_sharpened, agreement, p = gradient_sharpen(thermal, pan, sigma_vis=1.0)

    print(f"\nScale correction: p[0]={p[0]:.4f}  p[1]={p[1]:.4f}")
    print(f"Sharpened range:  {I_sharpened.min():.4f} to {I_sharpened.max():.4f}")

    # -------------------------------------------------------------------------
    #  Radiometric Evaluation
    # -------------------------------------------------------------------------
    thermal_celsius   = normalized_to_celsius(thermal,
                                               THERMAL_MIN_RAW, THERMAL_MAX_RAW)
    sharpened_celsius = normalized_to_celsius(I_sharpened,
                                               THERMAL_MIN_RAW, THERMAL_MAX_RAW)
    err_celsius = np.abs(thermal_celsius - sharpened_celsius)

    print(f"\nRadiometric Error:")
    print(f"  RMS error:  {np.sqrt(np.mean(err_celsius**2)):.2f}°C")
    print(f"  Max error:  {err_celsius.max():.2f}°C")
    print(f"  Mean error: {err_celsius.mean():.2f}°C")

    # Forehead ROI [x, y, w, h]
    roi = [1789, 422, 137, 92]
    x, y, rw, rh = roi
    roi_thermal   = thermal_celsius[y:y+rh, x:x+rw]
    roi_sharpened = sharpened_celsius[y:y+rh, x:x+rw]
    drift = abs(roi_thermal.mean() - roi_sharpened.mean())
    print(f"\nForehead ROI:")
    print(f"  Original:  {roi_thermal.mean():.2f}°C")
    print(f"  Sharpened: {roi_sharpened.mean():.2f}°C")
    print(f"  Drift:     {drift:.2f}°C")

    # Interior error (exclude boundary artifacts)
    border       = 50
    interior_err = err_celsius[border:-border, border:-border]
    print(f"\nInterior error (excluding {border}px border):")
    print(f"  RMS: {np.sqrt(np.mean(interior_err**2)):.2f}°C")
    print(f"  Max: {interior_err.max():.2f}°C")

    # -------------------------------------------------------------------------
    #  Save Results
    # -------------------------------------------------------------------------
    print("\nSaving results...")
    save_gray("_imgs/gs_thermal_original.png",  thermal)
    save_gray("_imgs/gs_thermal_sharpened.png", I_sharpened)
    save_gray("_imgs/gs_pan.png",               pan / pan.max())
    save_heatmap("_imgs/gs_agreement_map.png",  agreement)
    save_heatmap("_imgs/gs_error_celsius.png",  err_celsius)
    save_diverging("_imgs/gs_difference.png",   I_sharpened - thermal)

    # Save sharpened thermal as tiff for downstream evaluation
    tifffile.imwrite("_imgs/gs_thermal_sharpened.tiff", I_sharpened)

    print("  Saved to _imgs/gs_*.png")
    print("  Saved _imgs/gs_thermal_sharpened.tiff for evaluation")

    print("\n" + "=" * 70)
    print("[SUCCESS] Gradient Sharpening Test Complete")
    print("=" * 70)
