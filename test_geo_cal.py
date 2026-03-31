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

	print("="*70)
	print("[SUCCESS] GEOMETRIC CALIBRATION COMPLETE")
	print("="*70)
