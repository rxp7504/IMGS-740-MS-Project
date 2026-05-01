from flask import Flask, render_template, redirect, url_for
import cv2
import numpy as np
import os
import time
from pi_camera import RGBCamera
from thermal_camera import ThermalCamera

app = Flask(__name__)

# Folder to save captured images
CAPTURE_DIR = "static/captures"
os.makedirs(CAPTURE_DIR,exist_ok=True)

# Store the most recent capture filenames
last_capture = {"rgb": None, "thermal": None}

# Initialize cameras
rgb_cam = RGBCamera()
rgb_cam.start(mode="still")
thermal_cam = ThermalCamera()
thermal_cam.start(mode="radiometry")

print("Cameras Initialized")

@app.route("/")
def index():
	""" Main page - show capture button and images."""
	return render_template("index.html", images=last_capture)
	
@app.route("/capture",methods=["POST"])
def capture():
	""" Capture images from both cameras and save to disk"""
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	
	# --- RGB Camera ---
	rgb_frame = rgb_cam.capture()	
	rgb_filename = f"rgb_{timestamp}.jpg"
	cv2.imwrite(os.path.join(CAPTURE_DIR,rgb_filename),rgb_frame)
	
	# --- Thermal Camera ---
	thermal_frame = thermal_cam.capture()
	thermal_frame = cv2.rotate(thermal_frame,cv2.ROTATE_180)
	temp = (cv2.normalize(thermal_frame.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX) * 255).astype(np.uint8)
	thermal_8bit = cv2.applyColorMap(temp,cv2.COLORMAP_INFERNO)

	
	thermal_filename = f"thermal_{timestamp}.tiff"
	thermal_filename_8bit = f"thermal_{timestamp}_8bit.jpg"
	cv2.imwrite(os.path.join(CAPTURE_DIR,thermal_filename),thermal_frame)
	cv2.imwrite(os.path.join(CAPTURE_DIR,thermal_filename_8bit),thermal_8bit)

	print(f"Saved: {rgb_filename}, {thermal_filename}")
	
	# Update last capture
	last_capture["rgb"] = rgb_filename
	last_capture["thermal"] = thermal_filename_8bit
	
	# Redirect back to main page to show images
	return redirect(url_for("index"))
	
	
if __name__ == "__main__":
	try:
		app.run(host="0.0.0.0",port=5000,debug=False)
	finally:
		rgb_cam.stop()
		thermal_cam.stop()
	
