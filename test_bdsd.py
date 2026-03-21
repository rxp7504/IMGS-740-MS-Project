import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils


if __name__ == "__main__":

	print("="*70)
	print("[TEST] BDSD PANSHARPENING TEST")
	print("="*70)
	print("\n")

	# Parameters
	shape_hires = [480,640] # target thermal resolution
	shape_lores = [120,160] # starting thermal resolution

	# Import images
	rgb = cv2.imread("_dataset/_ADAS_1_3/FLIR_00006_rgb.jpg").astype(np.float32) / 255.0
	thermal_raw = cv2.imread("_dataset/_ADAS_1_3/FLIR_00006_16bit.tiff",cv2.IMREAD_UNCHANGED) # raw thermal temperature image
	print("[ORIGINAL RGB IMAGE]")     
	print(f"    Image shape: {rgb.shape}")
	print(f"    Image dtype: {rgb.dtype}")
	print(f"    Min pixel value: {np.min(rgb)}")
	print(f"    Max pixel value: {np.max(rgb)}")
	print("[ORIGINAL THERMAL IMAGE]")     
	print(f"    Image shape: {thermal_raw.shape}")
	print(f"    Image dtype: {thermal_raw.dtype}")
	print(f"    Min pixel value: {np.min(thermal_raw)}")
	print(f"    Max pixel value: {np.max(thermal_raw)}")

	print("[PREPARE IMAGES FOR PANSHARPENING]")
	PS_MS_HR_p, pan, raw_min, raw_max = utils.prepare_pansharp(
		thermal_raw=thermal_raw,
		rgb=rgb,
		warp_matrix_path="_resources/adas_warp_matrix.npy",
		shape_hires=(480, 640),
		shape_lores=(120, 160),
		verbose=True
	)

	print("\n")
	print("[SUCCESS] BDSD Test Complete")
