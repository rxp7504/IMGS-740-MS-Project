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

	# Resize the rgb to match the hi-res thermal
	thermal_h, thermal_w = thermal.shape[:2]
	target_size = (thermal_w * ratio, thermal_h * ratio)  # (width, height)
	rgb = utils.center_crop_to_aspect(rgb, thermal_w / thermal_h)
	rgb = cv2.resize(rgb, target_size)
	print("[AFTER RESIZING]")       
	print(f"    RGB shape: {rgb.shape}")
	print(f"    Thermal shape: {thermal.shape}")

	# Display images
	"""
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
	ax[0].set_title(f"RGB Image {rgb.shape}")
	ax[0].axis('off')
	ax[1].imshow(cv2.cvtColor(thermal_color,cv2.COLOR_BGR2RGB))
	ax[1].set_title(f"Thermal Image - Falsecolor {thermal_color.shape}")
	ax[1].axis('off')
	plt.tight_layout()
	plt.show()
	"""
	
	# Create the pseudo-multispectral low-resolution image
	PS_MS_LR = np.zeros((thermal_h,thermal_w,4),dtype=np.float32)
	PS_MS_LR[:,:,0:3] = (thermal_color / 255.0).astype(np.float32)
	PS_MS_LR[:,:,3] = thermal
	
	# Upsample to create the pseudo-multispectral high-resolution image
	PS_MS_HR_p = cv2.resize(PS_MS_LR,target_size)
	print(f"[PS-MS-HR'] Image Shape: {PS_MS_HR_p.shape}")
	
	#warp_matrix = utils.create_registration_matrix(PS_MS_HR_p[:,:,0:3], rgb, "_resources/warp_matrix.npy")
	# Apply registration matrix to rgb
	rgb_aligned = cv2.warpAffine(
		rgb,
		warp_matrix,
		(rgb.shape[1], rgb.shape[0]),
		flags=cv2.INTER_LINEAR
	)

	# Display registered images
	"""
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(cv2.cvtColor(rgb_aligned,cv2.COLOR_BGR2RGB))
	ax[0].set_title(f"RGB Image - Registered")
	ax[0].axis('off')
	ax[1].imshow(cv2.cvtColor(thermal_color,cv2.COLOR_BGR2RGB))
	ax[1].set_title(f"Thermal Image")
	ax[1].axis('off')
	plt.tight_layout()
	plt.show()
	"""
	
	# False color overlay: RGB in green channel, thermal in red channel
	utils.overlay_imgs(rgb_aligned,PS_MS_HR_p[:,:,0:3])
	
	# Create the "pan" image by greyscaling the registered RGB
	pan = cv2.cvtColor((rgb_aligned * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
	print(f"[PAN] Image Shape: {pan.shape}")

	print("\n")
	print("="*70)
	print("[SUCCESS] Prepare Image Test Complete")
	print("="*70)

