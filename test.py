from pi_camera import RGBCamera
from thermal_camera import ThermalCamera
import cv2
import numpy as np
import time
from picamera2 import Picamera2

with RGBCamera() as cam:
	# init.
	cam.start(mode="still")
	
	# capture and save an image
	frame = cam.capture()
	print("RGB Frame Shape: ",frame.shape)
	#frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	cv2.imwrite("_imgs/rgb_test.jpg",frame)
	print("RGB image saved")
	
	# print the sensor properties
	rgb_props = cam.get_properties()
	print("RGB Sensor Model: ",rgb_props["Model"])

with ThermalCamera() as flir:
	# init.
	flir.start("standard")
	
	# capture and save an image
	frame = flir.capture()
	print("IR Frame Shape: ",frame.shape)
	cv2.imwrite("_imgs/thermal_img.jpg",frame)
	print("Thermal image saved")
	
	# convert the raw image to temperature units
	frame_c = flir.convert_raw(frame) # Celsius
	frame_f = flir.convert_raw(frame,"f") # Fahrenheit
	
	# provide average scene temperature
	avg_c = round(float(np.mean(frame_c)), 2)
	print(f"Average scene temperature: {avg_c} °C")
	avg_f = round(float(np.mean(frame_f)), 2)
	print(f"Average scene temperature: {avg_f} °F")
	
	# Get sensor properties
	flir.get_properties()
	
