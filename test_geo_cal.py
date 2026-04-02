import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Geometric calibration on the thermal camera to correct lens distortion
"""

if __name__ == "__main__":
	
	print("="*70)
	print("TEST THERMAL GEOMETRIC CALIBRATION")
	print("="*70)
	
	# import thermal image
	thermal = cv2.imread("_cal_imgs/thermal_cal_2.tiff",cv2.IMREAD_UNCHANGED)
	thermal_8bit = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	
	# calibration target dimensions
	board_w = 1485.9 # [mm]
	board_h = 1066.8 # [mm]
	
	# number of points on target (handwarmers)
	pts_w = 5
	pts_h = 4
	
	# calculate spacing between points
	v_spacing = board_h / (pts_h + 1) # [mm]
	h_spacing = board_w / (pts_w + 1) # [mm]
	
	objp = np.array([
		# Row 0
		[0,           0,           0],
		[h_spacing,   0,           0],
		[h_spacing*2, 0,           0],
		[h_spacing*3, 0,           0],
		[h_spacing*4, 0,           0],
		# Row 1
		[0,           v_spacing,   0],
		[h_spacing,   v_spacing,   0],
		[h_spacing*2, v_spacing,   0],
		[h_spacing*3, v_spacing,   0],
		[h_spacing*4, v_spacing,   0],
		# Row 2
		[0,           v_spacing*2, 0],
		[h_spacing,   v_spacing*2, 0],
		[h_spacing*2, v_spacing*2, 0],
		[h_spacing*3, v_spacing*2, 0],
		[h_spacing*4, v_spacing*2, 0],
		# Row 3
		[0,           v_spacing*3, 0],
		[h_spacing,   v_spacing*3, 0],
		[h_spacing*2, v_spacing*3, 0],
		[h_spacing*3, v_spacing*3, 0],
		[h_spacing*4, v_spacing*3, 0],
	], dtype=np.float32)

	# threshold to isolate hot handwarmers
	_, thresh = cv2.threshold(thermal_8bit,100,255,cv2.THRESH_BINARY)
	
	# find contours of each handwarmer
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	centroids = []
	for c in contours:
		M = cv2.moments(c)
		if M["m00"] > 0:
			cx = M["m10"] / M["m00"]
			cy = M["m01"] / M["m00"]
			centroids.append((cx,cy))
			
	centroids = sorted(centroids, key=lambda p: (round(p[1]/20)*20,p[0])) # sort row by row
	print(f"found {len(centroids)} centroids")
	
	# visualize to verify
	display = cv2.cvtColor(thermal_8bit,cv2.COLOR_GRAY2BGR)
	for i, (cx,cy) in enumerate(centroids):
		cv2.circle(display,(int(cx),int(cy)),3,(0,255,0),-1)
		cv2.putText(display,str(i),(int(cx)+4,int(cy)-4),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
		
	plt.imshow(display)
	plt.show()
	
	img_points = np.array(centroids, dtype=np.float32).reshape(-1, 1, 2)
	obj_points = objp.reshape(-1, 1, 3)

	#ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
	#	[obj_points], [img_points],
	#	(160, 120),  # thermal image size
	#	None, None
	#)
	
	# Lepton 3.5 known FOV
	fov_h = 57.0  # degrees
	fov_v = 44.0  # degrees
	w, h = 160, 120

	fx = (w / 2) / np.tan(np.radians(fov_h / 2))
	fy = (h / 2) / np.tan(np.radians(fov_v / 2))
	cx, cy = w / 2.0, h / 2.0

	K = np.array([
		[fx,  0,  cx],
		[0,  fy,  cy],
		[0,   0,   1]
	], dtype=np.float32)

	print(f"Calculated focal lengths: fx={fx:.2f}, fy={fy:.2f}")

	# Fix focal length and principal point, only solve for distortion
	flags = (cv2.CALIB_USE_INTRINSIC_GUESS + 
			 cv2.CALIB_FIX_FOCAL_LENGTH + 
			 cv2.CALIB_FIX_PRINCIPAL_POINT)

	ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
		[obj_points], [img_points],
		(w, h), K, None,
		flags=flags
	)

	print(f"RMS reprojection error: {ret:.4f}")
	print(f"Camera matrix K:\n{K}")
	print(f"Distortion coefficients: {dist}")

	#print(f"RMS reprojection error: {ret:.4f}")  # should be < 1.0
	#print(f"Camera matrix K:\n{K}")
	#print(f"Distortion coefficients: {dist}")

	# Save calibration
	np.save("_resources/thermal_K.npy", K)
	np.save("_resources/thermal_dist.npy", dist)
	
	thermal_undistorted = cv2.undistort(thermal_8bit, K, dist)
	#cv2.imwrite("_imgs/thermal_undistorted.png", thermal_undistorted)
	plt.imshow(thermal_undistorted)
	plt.show()
	

	print("\n")
	print("="*70)
	print("[SUCCESS] GEOMETRIC CALIBRATION COMPLETE")
	print("="*70)
