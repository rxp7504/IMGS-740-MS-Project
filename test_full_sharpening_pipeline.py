import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

"""
Test script for the full thermal image sharpening pipeline.
"""


def undistort_thermal(thermal_raw,K,dist):
	return cv2.undistort(cv2.normalize(thermal_raw, None, 0.0, 1.0, cv2.NORM_MINMAX),K,dist)

if __name__ == "__main__":
	print("="*70)
	print("[TEST] FULL THERMAL SHARPENING PIPELINE TEST")
	print("="*70)
	print("\n")
	
    # -------------------------------------------------------------------------
    #  Load Data
    # -------------------------------------------------------------------------
	
	# Timestamp of image pair to load
	img_tstamp = "20260420_192034"
	ratio = 20 # how many times to boost the thermal resolution
	
	# Load thermal image
	thermal_path = f"static/captures/thermal_{img_tstamp}.tiff"
	thermal_raw = cv2.imread(thermal_path,cv2.IMREAD_UNCHANGED).astype(np.float32) # captured radiometric flir leption image

	# Load RGB image
	rgb_path = f"static/captures/rgb_{img_tstamp}.jpg"
	rgb = cv2.imread(rgb_path).astype(np.float32) / 255.0 # captured raspberry pi rgb image
	
	# Load distortion correction
	K_path = "_resources/thermal_K.npy"
	K = np.load(K_path)
	dist_path = "_resources/thermal_dist.npy"
	dist = np.load(dist_path)
	
	# Load homography registration matrix
	H = np.load("_resources/H_20x.npy")	
	
    # -------------------------------------------------------------------------
    #  Image Pre-processing
    # -------------------------------------------------------------------------
	
	# Store the min/max raw thermal image values
	THERMAL_MIN_RAW = float(thermal_raw.min())
	THERMAL_MAX_RAW = float(thermal_raw.max())	
	
	# Undistort and normalize the thermal image
	thermal = undistort_thermal(thermal_raw,K,dist)
	cv2.imwrite("_imgs/thermal_truth.tiff",thermal) # Write out the undistorted thermal image
	
	# Convert the images to psudo-multi spectral and panchromatic
	PS_MS_HR_p, pan = utils.prepare_pansharp(rgb,thermal,H,ratio,verbose=True)


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
    #save_heatmap("_imgs/gs_agreement_map.png",  agreement)
    #save_heatmap("_imgs/gs_error_celsius.png",  err_celsius)
    #save_diverging("_imgs/gs_difference.png",   I_sharpened - thermal)

    # Save sharpened thermal as tiff for downstream evaluation
    tifffile.imwrite("_imgs/gs_thermal_sharpened.tiff", I_sharpened)

    print("  Saved to _imgs/gs_*.png")
    print("  Saved _imgs/gs_thermal_sharpened.tiff for evaluation")

    print("\n" + "=" * 70)
    print("[SUCCESS] Gradient Sharpening Test Complete")
    print("=" * 70)
