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
last_capture = {"rgb": None, "thermal", None}

@app.route("/")
def index():
	""" Main page - show capture button and images."""
	return render_template("index.html", images=last_capture)
	
@app.route("/capture",methods=["POST"])
def capture():
	""" Capture images from both cameras and save to disk"""
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	
	# --- RGB Camera ---
	with RGBCamera() as rgb_cam:
		rgb_cam.start(mode="still")
		rgb_frame = rgb_cam.capture()
		
	rgb_filename = f"rgb_{timestamp}.jpg"
	cv2.imwrite(os.path.join(CAPTURE_DIR,rgb_filename),rgb_frame)
	
	# --- Thermal Camera ---
	with ThermalCamera() as thermal_cam:
		thermal_cam.start()
		thermal_frame = thermal_cam.capture()
		
	thermal_filename = f"thermal_{timestamp}.jpg"
	cv2.imwrite(os.path.join(CAPTURE_DIR,thermal_filename),thermal_frame)
	
	print(f"Saved: {rgb_filename}, {thermal_filename}")
	
	# Update last capture
	last_capture["rgb"] = rgb_filename
	last_capture["thermal"] = thermal_filename
	
	# Redirect back to main page to show images
	return redirect(url_for("index"))
	
	
if __name__ == "__main__":
	app.run(host="0.0.0.0",port=500,debug=True)
	
	
