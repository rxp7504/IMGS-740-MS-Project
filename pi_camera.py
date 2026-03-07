import cv2
import numpy as np
import time
from picamera2 import Picamera2


class RGBCamera:
	def __init__(self, resolution=(3280,2464), preview_resolution=(1920, 1080)):
		"""
		Initialize the Raspberry Pi RGB camera.

		Args:
		resolution: Full sensor resolution for still captures (default: max 12MP)
		preview_resolution: Lower resolution for fast/preview captures
		"""
		self.resolution = resolution
		self.preview_resolution = preview_resolution
		self.picam2 = Picamera2()
		self.properties = self.picam2.camera_properties
		self._configured = False
		self._started = False

	def start(self, mode="still"):
		"""
		Configure and start the camera.

		Args:
		mode: "still" for high-res captures, "preview" for fast/video captures
		"""
		if mode == "still":
			cfg = self.picam2.create_still_configuration(
			main={"format": "RGB888", "size": self.resolution}
			)
		elif mode == "preview":
			cfg = self.picam2.create_preview_configuration(
			main={"format": "RGB888", "size": self.preview_resolution}
			)
		else:
			raise ValueError(f"Unknown mode '{mode}'. Use 'still' or 'preview'.")

		self.picam2.configure(cfg)
		self.picam2.start()
		self._started = True
		self._configured = True
		time.sleep(1)  # warm up
		print(f"RGB camera started in '{mode}' mode")

	def capture(self):
		"""
		Capture a single frame and return it as a BGR numpy array (OpenCV format).

		Returns:
		numpy array (H, W, 3) in BGR format
		"""
		if not self._started:
			raise RuntimeError("Camera not started. Call start() first.")

		frame = self.picam2.capture_array("main")  # RGB888
		frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		return frame_bgr

	def get_properties(self):
		return self.properties

	def stop(self):
		"""Stop and release the camera."""
		if self._started:
			self.picam2.stop()
			self._started = False
			print("RGB camera stopped")

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()

