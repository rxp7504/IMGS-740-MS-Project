from pi_camera import RGBCamera
from thermal_camera import ThermalCamera
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from picamera2 import Picamera2

print("="*70)
print("\n[TEST] TESTING CAMERA CONFIG/CAPTURE AND IMAGE SAVE\n")
print("="*70)
print("\n")

print("="*70)
print("[RGB CAMERA TEST]")
print("="*70)
with RGBCamera() as cam:
	# init.
	cam.start(mode="still")
	print("		\nRGB Camera Initialized")
	
	# print the sensor properties
	rgb_props = cam.get_properties()
	print("		RGB Sensor Model: ",rgb_props["Model"])
	
	# set white balance
	print("		Setting white balance settings...")
	cam.set_whitebalance()
	time.sleep(3)
	metadata = cam.picam2.capture_metadata()
	print("Color gains:", metadata.get("ColourGains"))
	print("AWB mode:", metadata.get("AwbMode"))
	
	# capture and save an image
	frame = cam.capture()
	print("		RGB image captured")
	print("		RGB Frame Shape: ",frame.shape)
	rgb_out = "_imgs/rgb.jpg"
	cv2.imwrite(rgb_out,frame)
	print(f"		RGB image saved: {rgb_out}")
	
	"""
	# plot histogram of the image
	colors = ('b','g','r')
	labels = ('Blue','Green','Red')
	plt.figure(figsize=(10,4))
	plt.title("Image Histogram")
	for i, (color,label) in enumerate(zip(colors,labels)):
		hist = cv2.calcHist([frame],[i],None,[256],[0,256])
		plt.plot(hist,color=color,label=label)
	plt.xlabel("Pixel Value")
	plt.ylabel("Frequency")
	plt.legend()
	plt.tight_layout()
	plt.savefig("_imgs/hist.png")
	plt.show()
	"""
	

print("\n")
print("="*70)
print("[THERMAL CAMERA TEST]")
print("="*70)
with ThermalCamera() as flir:
	# init.
	flir.start("radiometry")
	print("		IR camera initialized")
	
	# capture and save an image
	frame = flir.capture()
	print("		IR image captured")
	print("		IR Frame Shape: ",frame.shape)
	ir_out = "_imgs/thermal.tiff"
	cv2.imwrite(ir_out,frame)
	print(f"		IR image saved: {ir_out}")
	
	# convert the raw image to temperature units
	frame_c = flir.convert_raw(frame) # Celsius
	frame_f = flir.convert_raw(frame,"f") # Fahrenheit
	
	# provide average scene temperature
	avg_c = round(float(np.mean(frame_c)), 2)
	print(f"	Average scene temperature: {avg_c} °C")
	avg_f = round(float(np.mean(frame_f)), 2)
	print(f"	Average scene temperature: {avg_f} °F")
	
	# Get sensor properties
	flir.get_properties()
	
print("="*70)
print("\n [SUCCESS] Camera Test Complete")
	
