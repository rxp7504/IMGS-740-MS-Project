import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

"""
Test script for preparing captured images for pansharpening.

Input format should be BGR float32 normalized by bit depth (0-1)
"""

if __name__ == "__main__":
	print("="*70)
	print("[TEST] PREPARE CAPTURED IMAGES FOR PANSHARPENING TEST")
	print("="*70)
	print("\n")
	
	thermal_raw = cv2.imread("_imgs/thermal.tiff",cv2.IMREAD_UNCHANGED).astype(np.float32) # captured radiometric flir leption image
	print(f"[RAW THERMAL IMAGE]")
	print(f"    Min pixel value: {np.min(thermal_raw)}")
	print(f"    Max pixel value: {np.max(thermal_raw)}")
	thermal_min = float(thermal_raw.min())
	thermal_max = float(thermal_raw.max())
	
	# Input Parameters
	rgb = cv2.imread("_imgs/rgb.jpg").astype(np.float32) / 255.0 # captured raspberry pi rgb image
	thermal = cv2.rotate(cv2.normalize(thermal_raw, None, 0.0, 1.0, cv2.NORM_MINMAX),cv2.ROTATE_180)
	warp_matrix = np.load("_resources/warp_matrix.npy")
	ratio = 4 # how many times to boost the thermal resolution
	
	# Convert the images
	PS_MS_HR_p, pan = utils.prepare_pansharp(rgb,thermal,warp_matrix,ratio,verbose=True)

	# False color overlay: RGB in green channel, thermal in red channel
	pan_rgb = np.stack([pan]*3,axis=-1)
	utils.overlay_imgs(pan_rgb,PS_MS_HR_p[:,:,0:3])
	
	# Save the images 
	cv2.imwrite("_imgs/pan.tiff",pan)
	cv2.imwrite("_imgs/PS_MS_HR_p.tiff",PS_MS_HR_p)
	print("\n Images saved")

	print("\n")
	print("="*70)
	print("[SUCCESS] Prepare Image Test Complete")
	print("="*70)

