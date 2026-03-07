from pi_camera import RGBCamera
import cv2
import numpy as np
import time
from picamera2 import Picamera2

with RGBCamera() as rgbCam:
	# init.
	rgbCam.start(mode="still")
	
	# capture and save an image
	#frame = rgbCam.capture()
	#cv2.imwrite("rgb_test.jpg",frame)
	
	# print the sensor properties
	rgb_props = rgbCam.get_properties()
	print("RGB Sensor Model: ",rgb_props["Model"])
