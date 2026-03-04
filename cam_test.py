from picamera2 import Picamera2
import cv2
import numpy as np
import time

picam2 = Picamera2() # create camera object

#picam2.start_and_capture_file("test.jpg")
#print("finished")

cfg = picam2.create_still_configuration(main={"format": "RGB888"}) # create the config for hi-res images
#cfg = picam2.create_preview_configuration() # create the config for preview images

picam2.configure(cfg) # load the config
picam2.start()
#picam2.start_preview(Preview.QTGL)
#picam2.start()
#time.sleep(2)
#picam2.capture_file("test.jpg")
#print("finished")

time.sleep(1)
frame = picam2.capture_array("main")

cv2.imwrite("test.jpg",frame)
print("image saved!")

