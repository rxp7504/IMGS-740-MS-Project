import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils

def crop_to_size(img, shape_hires):
	# Calculate the total pixels to remove
	diff_h = img.shape[0] - shape_hires[0]
	diff_w = img.shape[1] - shape_hires[1]

	# Use floor division (//) to get integer start/end points
	# This handles the "half" distance from each side
	start_h = diff_h // 2
	start_w = diff_w // 2

	# Slice using the start point and the desired final shape
	img_new = img[start_h : start_h + shape_hires[0], 
				  start_w : start_w + shape_hires[1]]

	return img_new

if __name__ == "__main__":
	print("="*70)
	print("[TEST] PREPARE IMAGE FOR PANSHARPENING TEST")
	print("="*70)
	print("\n")

	# Parameters
	shape_hires = [480,640] # target thermal resolution
	shape_lores = [120,160] # starting thermal resolution
	ratio = 4 # max ratio between rgb and thermal images

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

	# Crop images to fit the same aspect ratio
	thermal_raw_cropped = crop_to_size(thermal_raw.astype(np.float32),shape_hires)
	raw_min = float(thermal_raw_cropped.min())
	raw_max = float(thermal_raw_cropped.max())
	thermal = cv2.normalize(thermal_raw_cropped.astype(np.float32),None,0.0,1.0,cv2.NORM_MINMAX) # normalize the thermal image 0-1

	target_h = int(rgb.shape[1] * (shape_hires[0] / shape_hires[1]))
	target_shape = [target_h, rgb.shape[1]]
	
	rgb = crop_to_size(rgb, target_shape)
	
	# Create a false color thermal image
	thermal_color = cv2.applyColorMap((thermal * 255).astype(np.uint8),cv2.COLORMAP_INFERNO)

	# Resize the rgb to math the hi-res thermal
	rgb = cv2.resize(rgb,(shape_hires[1],shape_hires[0]))
	print("[AFTER RESIZING]")       
	print(f"    RGB shape: {rgb.shape}")
	print(f"    Thermal shape: {thermal.shape}")

	# Display images
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
	ax[0].set_title(f"RGB Image {rgb.shape}")
	ax[0].axis('off')
	ax[1].imshow(cv2.cvtColor(thermal_color,cv2.COLOR_BGR2RGB))
	ax[1].set_title(f"Thermal Image - Falsecolor {thermal_color.shape}")
	ax[1].axis('off')
	plt.tight_layout()
	plt.show()


	# Manual point selection
	#warp_matrix = utils.create_registration_matrix(cv2.cvtColor(utils.contrast_enhance(thermal),cv2.COLOR_GRAY2BGR), rgb, "_resources/adas_warp_matrix2.npy")
	warp_matrix = np.load("_resources/adas_warp_matrix.npy")

	# Apply to rgb
	rgb_aligned = cv2.warpAffine(
		rgb,
		warp_matrix,
		(rgb.shape[1], rgb.shape[0]),
		flags=cv2.INTER_LINEAR
	)

	# Display registered images
	fig, ax = plt.subplots(1,2)
	ax[0].imshow(cv2.cvtColor(rgb_aligned,cv2.COLOR_BGR2RGB))
	ax[0].set_title(f"RGB Image - Registered")
	ax[0].axis('off')
	ax[1].imshow(cv2.cvtColor(thermal_color,cv2.COLOR_BGR2RGB))
	ax[1].set_title(f"Thermal Image")
	ax[1].axis('off')
	plt.tight_layout()
	plt.show()

	# False color overlay: RGB in green channel, thermal in red channel
	utils.overlay_imgs(rgb_aligned,thermal_color)

	# Create the "pan" image by greyscaling the registered RGB
	pan = cv2.cvtColor((rgb_aligned * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
	print(f"[PAN] Image Shape: {pan.shape}")
	
	# Create the "Pseudo-MultiSpectral" image
	PS_MS_HR = np.zeros((shape_hires[0],shape_hires[1],4),dtype=np.float32)
	PS_MS_HR[:,:,0:3] = (thermal_color / 255.0).astype(np.float32)
	PS_MS_HR[:,:,3] = thermal
	
	# Downsample PS-MS image to low-res (this simulates our incoming image)
	PS_MS_LR = cv2.resize(PS_MS_HR,shape_lores)
	print(f"[PS-MS-LR] Image Shape: {PS_MS_LR.shape}")
	
	# Upsample the lowres image
	PS_MS_HR_p = cv2.resize(PS_MS_LR,shape_hires)
	print(f"[PS-MS-HR'] Image Shape: {PS_MS_HR_p.shape}")

	
	print("\n")
	print("[SUCCESS] BDSD Test Complete")

