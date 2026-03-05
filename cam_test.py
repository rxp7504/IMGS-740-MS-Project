# Testing the Raspberry Pi Camera
from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Init. the pi camera
picam2 = Picamera2() # create camera object
cfg = picam2.create_still_configuration(main={"format": "RGB888"}) # create the config for hi-res images in BGR format
picam2.configure(cfg) # load the config
picam2.start() # start the camera
print("Pi camera initialized")

# Capture an image
time.sleep(1) # wait for a second
frame = picam2.capture_array("main") # capture an array
print("Frame Shape (h,w,ch): ",frame.shape) # print the shape

# Save the image
cv2.imwrite("test.jpg",frame)
print("image saved!")

