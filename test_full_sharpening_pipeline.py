import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import time
import tifffile



"""
Test script for the full thermal image sharpening pipeline.
"""


if __name__ == "__main__":
	print("="*70)
	print("[TEST] FULL THERMAL SHARPENING PIPELINE TEST")
	print("="*70)
	print("\n")
	
    # -------------------------------------------------------------------------
    #  Load Data
    # -------------------------------------------------------------------------
	
	img_tstamp = "20260420_192034" # timestamp of image pair to load
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
    #  Gradient sharpening
    # -------------------------------------------------------------------------
	start = time.time()

	# Run the whole thermal sharpening algorithm
	thermal_sharpened, pan, thermal_sharpened_raw, thermal_upscale = utils.gradient_sharpen_thermal(thermal_raw,rgb,ratio,K,dist,H)

	elapsed = time.time() - start
	print(f"Elapsed time: {elapsed:.2f}s")
    # -------------------------------------------------------------------------
    #  Radiometric Evaluation
    # -------------------------------------------------------------------------

	thermal_celsius   = utils.raw_to_celsius(thermal_raw)
	sharpened_celsius = utils.raw_to_celsius(cv2.resize(thermal_sharpened_raw, (thermal_raw.shape[1], thermal_raw.shape[0])))
	err_celsius = np.abs(thermal_celsius - sharpened_celsius)

	print(f"\nRadiometric Error:")
	print(f"  RMS error:  {np.sqrt(np.mean(err_celsius**2)):.2f}°C")
	print(f"  Max error:  {err_celsius.max():.2f}°C")
	print(f"  Mean error: {err_celsius.mean():.2f}°C")

	# Interior error (exclude boundary artifacts)
	border       = 10
	interior_err = err_celsius[border:-border, border:-border]
	print(f"\nInterior error (excluding {border}px border):")
	print(f"  RMS: {np.sqrt(np.mean(interior_err**2)):.2f}°C")
	print(f"  Max: {interior_err.max():.2f}°C")

    # -------------------------------------------------------------------------
    #  Save Results
    # -------------------------------------------------------------------------
	print("\nSaving results...")
	#cv2.imwrite("_imgs/thermal_truth.tiff",thermal) # Write out the undistorted thermal image
	utils.save_gray("_imgs/gs_thermal_original.png",  thermal_upscale)
	utils.save_gray("_imgs/gs_thermal_sharpened.png", thermal_sharpened)
	utils.save_gray("_imgs/gs_pan.png",               pan / pan.max())
	#save_heatmap("_imgs/gs_agreement_map.png",  agreement)
	#save_heatmap("_imgs/gs_error_celsius.png",  err_celsius)
	#save_diverging("_imgs/gs_difference.png",   I_sharpened - thermal)

	# Save sharpened thermal as tiff for downstream evaluation
	tifffile.imwrite("_imgs/gs_thermal_sharpened.tiff", thermal_sharpened_raw)

	print("  Saved to _imgs/gs_*.png")
	print("  Saved _imgs/gs_thermal_sharpened.tiff for evaluation")

	print("\n" + "=" * 70)
	print("[SUCCESS] Gradient Sharpening Test Complete")
	print("=" * 70)
