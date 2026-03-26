import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils



if __name__ == "__main__":
	print("="*70)
	print("[FULLY TRANSFORM CAPTURED IMAGES TO PREPARE FOR PANSHARPENING]")
	print("="*70)
	
	# Import captured images
    rgb = cv2.imread("_imgs/rgb_test.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.imread("_imgs/thermal_img.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.rotate(thermal, cv2.ROTATE_180)  # thermal camera is upside down

	# Create registration matrix
	create_registration_matrix(img_fixed, img_moving, filename):

    # Load registration matrix
    warp_matrix = np.load("_resources/warp_matrix.npy")

