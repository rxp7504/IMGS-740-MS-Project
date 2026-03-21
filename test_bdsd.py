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
	
	from bdsd import bdsd

	result = bdsd(PS_MS_HR_p, pan, ratio=4, S=32)
	print(f"[BDSD OUTPUT] Shape: {result.shape}")

	# Split output bands
	false_color_enhanced = (result[:, :, 0:3] * 255).astype(np.uint8)
	thermal_enhanced     = result[:, :, 3]

	# Denormalize thermal back to raw counts
	thermal_enhanced_raw = (thermal_enhanced * (raw_max - raw_min)) + raw_min

	# Convert to Celsius if using Lepton (skip for ADAS dataset)
	# thermal_enhanced_celsius = (thermal_enhanced_raw / 100.0) - 273.15

	# Display
	fig, ax = plt.subplots(1, 3, figsize=(15, 5))
	ax[0].imshow(cv2.cvtColor((PS_MS_HR_p[:,:,0:3] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
	ax[0].set_title("PS-MS-HR' (upsampled input)")
	ax[0].axis('off')
	ax[1].imshow(cv2.cvtColor(false_color_enhanced, cv2.COLOR_BGR2RGB))
	ax[1].set_title("Enhanced False Color")
	ax[1].axis('off')
	ax[2].imshow(thermal_enhanced, cmap='inferno')
	ax[2].set_title("Enhanced Thermal")
	ax[2].axis('off')
	plt.tight_layout()
	plt.show()

	print("\n")
	print("[SUCCESS] BDSD Test Complete")
