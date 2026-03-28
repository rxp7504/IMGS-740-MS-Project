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
	#thermal_min = float(thermal_raw.min())
	#thermal_max = float(thermal_raw.max())
	
	# Input Parameters
	rgb = cv2.imread("_imgs/rgb.jpg").astype(np.float32) / 255.0 # captured raspberry pi rgb image
	thermal = thermal_raw / 65535.0
	ratio = 4 # how many times to boost the thermal resolution
	
	print("[ORIGINAL RGB IMAGE]")     
	print(f"    Image shape: {rgb.shape}")
	print(f"    Image dtype: {rgb.dtype}")
	print(f"    Min pixel value: {np.min(rgb)}")
	print(f"    Max pixel value: {np.max(rgb)}")
	print("[ORIGINAL THERMAL IMAGE]")     
	print(f"    Image shape: {thermal.shape}")
	print(f"    Image dtype: {thermal.dtype}")
	print(f"    Min pixel value: {np.min(thermal)}")
	print(f"    Max pixel value: {np.max(thermal)}")
	
	# Create a false color thermal image
	thermal_color = cv2.applyColorMap((thermal * 255).astype(np.uint8),cv2.COLORMAP_INFERNO)

	# Resize the rgb to math the hi-res thermal
	thermal_h, thermal_w = thermal.shape[:2]
	target_size = (thermal_w * ratio, thermal_h * ratio)  # (width, height)
	rgb = utils.center_crop_to_aspect(rgb, thermal_w / thermal_h)
	rgb = cv2.resize(rgb, target_size)

	print("[AFTER RESIZING]")       
	print(f"    RGB shape: {rgb.shape}")
	print(f"    Thermal shape: {thermal.shape}")
