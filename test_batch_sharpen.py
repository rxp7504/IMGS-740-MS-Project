"""
Batch Thermal Image Sharpening for Quantitative Evaluation
===========================================================

Evaluation protocol (Wald's protocol):
    1. Load full-res thermal (160x120) — saved as truth
    2. Load full-res RGB (3280x2464)
    3. Undistort thermal at full resolution
    4. Register RGB to thermal at full resolution (homography stays valid)
       → pan_full at 160x120
    5. Degrade ONLY thermal by deg_fac → 80x60
       Pan stays at 160x120 as the high-res guide (mirrors deployment)
    6. Run sharpening: 80x60 thermal + 160x120 pan → 160x120 output
    7. Save truth, bicubic baseline, and sharpened for sewar evaluation
"""

import cv2
import numpy as np
import utils
import time
import tifffile
import os
import re


def get_capture_timestamps(captures_dir: str) -> list:
    """
    Extract unique timestamps from capture folder.
    Files are named: thermal_YYYYMMDD_HHMMSS.tiff / rgb_YYYYMMDD_HHMMSS.jpg

    Returns sorted list of timestamp strings.
    """
    timestamps = set()
    pattern = re.compile(r'(?:thermal|rgb)_(\d{8}_\d{6})')

    for fname in os.listdir(captures_dir):
        match = pattern.match(fname)
        if match:
            timestamps.add(match.group(1))

    return sorted(timestamps)


def get_complete_timestamps(captures_dir: str) -> list:
    """Return only timestamps where both thermal and rgb files exist."""
    timestamps = get_capture_timestamps(captures_dir)
    complete = []
    for ts in timestamps:
        thermal_exists = os.path.exists(f"{captures_dir}/thermal_{ts}.tiff")
        rgb_exists     = os.path.exists(f"{captures_dir}/rgb_{ts}.jpg")
        if thermal_exists and rgb_exists:
            complete.append(ts)
    return complete


if __name__ == "__main__":
    print("=" * 70)
    print("[TEST] BATCH IMAGE SHARPENING")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    #  Configuration
    # -------------------------------------------------------------------------
    captures_dir = "/home/rxp7504/repo/IMGS-740-MS-Project/static/captures"
    out_dir      = "/media/rxp7504/RJ USB/eval_imgs"
    deg_fac      = 2   # degrade thermal and pan by this factor before sharpening
    ratio        = deg_fac  # recover back to original thermal resolution

    os.makedirs(out_dir, exist_ok=True)

    # Load calibration resources
    K    = np.load("_resources/thermal_K.npy")
    dist = np.load("_resources/thermal_dist.npy")
    H    = np.load("_resources/H_20x.npy")

    # -------------------------------------------------------------------------
    #  Process each capture set
    # -------------------------------------------------------------------------
    timestamps = get_complete_timestamps(captures_dir)
    print(f"Found {len(timestamps)} complete capture sets\n")

    for ts in timestamps:
        print(f"[{ts}]")

        # Load images
        thermal_raw = cv2.imread(f"{captures_dir}/thermal_{ts}.tiff",
                                 cv2.IMREAD_UNCHANGED).astype(np.float32)
        rgb         = cv2.imread(f"{captures_dir}/rgb_{ts}.jpg").astype(np.float32) / 255.0

        t_h, t_w = thermal_raw.shape[:2]
        print(f"  Thermal: {thermal_raw.shape}  RGB: {rgb.shape}")

        # Store radiometric range before any processing
        THERMAL_MIN_RAW = float(thermal_raw.min())
        THERMAL_MAX_RAW = float(thermal_raw.max())

        # ---------------------------------------------------------------------
        #  Step 1: Undistort and normalize at FULL resolution
        # ---------------------------------------------------------------------
        thermal_full = utils.undistort_thermal(thermal_raw, K, dist)  # 160x120, normalized 0-1

        # ---------------------------------------------------------------------
        #  Step 2: Register RGB using correct ratio (homography was computed
        #  for 20x upscaled thermal space). Resize down to 160x120 after.
        # ---------------------------------------------------------------------
        pan_hires = utils.prepare_pan(rgb, thermal_full, H, ratio=20)  # 3200x2400
        pan_full  = cv2.resize(pan_hires, (t_w, t_h), interpolation=cv2.INTER_AREA)
        del pan_hires

        # ---------------------------------------------------------------------
        #  Step 3: Degrade ONLY thermal — pan stays at full resolution
        #  thermal_deg: 80x60  (input to sharpening)
        #  pan_full:    160x120 (high-res guide — mirrors deployment)
        # ---------------------------------------------------------------------
        thermal_deg = cv2.resize(thermal_full, (t_w // deg_fac, t_h // deg_fac))
        print(f"  Degraded thermal: {thermal_deg.shape}  Pan guide: {pan_full.shape}")

        # Bicubic baseline — upsample degraded thermal back to original size
        thermal_bicubic = cv2.resize(thermal_deg, (t_w, t_h),
                                     interpolation=cv2.INTER_CUBIC)

        # ---------------------------------------------------------------------
        #  Step 4: Run gradient sharpening
        #  Upsample degraded thermal to match pan resolution first
        #  Input:  160x120 upscaled thermal + 160x120 pan (high-res guide)
        #  Output: 160x120 sharpened thermal (same as truth)
        # ---------------------------------------------------------------------
        # Upsample degraded thermal to match pan resolution
        thermal_upscaled = cv2.resize(thermal_deg, (t_w, t_h),
                                      interpolation=cv2.INTER_CUBIC)

        start = time.time()
        thermal_sharpened, agreement, p = utils.gradient_sharpen(
            thermal_upscaled, pan_full, sigma_vis=1.0)
        elapsed = time.time() - start

        # Convert sharpened back to raw Lepton units for saving
        thermal_sharpened_raw = thermal_sharpened * (THERMAL_MAX_RAW - THERMAL_MIN_RAW) \
                                + THERMAL_MIN_RAW

        print(f"  Sharpened shape:  {thermal_sharpened.shape}")
        print(f"  Scale correction: p[0]={p[0]:.4f}  p[1]={p[1]:.4f}")
        print(f"  Elapsed time:     {elapsed:.2f}s")

        # ---------------------------------------------------------------------
        #  Step 5: Save results for sewar evaluation
        #  All images at 160x120 (original thermal resolution)
        # ---------------------------------------------------------------------
        # Truth: original full-res thermal (ground truth for metrics)
        tifffile.imwrite(f"{out_dir}/truth_{ts}.tiff",    thermal_full)

        # Bicubic baseline
        tifffile.imwrite(f"{out_dir}/bicubic_{ts}.tiff",  thermal_bicubic)

        # Gradient sharpening result
        tifffile.imwrite(f"{out_dir}/gs_{ts}.tiff",       thermal_sharpened)

        # Pan image used as guide (full resolution, for reference)
        tifffile.imwrite(f"{out_dir}/pan_{ts}.tiff",      pan_full)

        print(f"  Saved to {out_dir}/\n")

    print("=" * 70)
    print("[SUCCESS] Batch Sharpening Complete")
    print("=" * 70)
