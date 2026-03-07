import cv2
import numpy as np

class ThermalCamera:
	def __init__(self,device_index=0):
		"""
		Initialize the FLIR Lepton thermal camera via PureThermal 3 board.
		
		Args:
			device_index: V4L2 device index (default: 0 for /dev.video0)
		"""
		self.device_index = device_index
		self.cap = None
		self._started = False
		
	def start(self):
		# Open and configure the thermal camera
		self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
		self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY'))
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
		self.cap.set(cv2.CAP_PROP_FPS, 9)
		
		if not self.cap.isOpened():
			raise RuntimeError(f"Could not open thermal camera at /dev/video{self.device_index}")
			
		# flush the buffer
		for _ in range(5):
			self.cap.grab()
			
		self._started = True
		print(f"Thermal camera started on /dev/video{self.device_index}")
			
	def capture(self):
		"""
		Capture a raw thermal frame in BGR format.
		
		Returns:
			numpy array (H, W, 3) in BGR format
		"""
		if not self._started:
			raise RuntimeError("Camera not started. Call start() first.")
			
		ret, frame = self.cap.read()
		if not ret:
			raise RuntimeError("Failed to capture thermal frame")
			
		return frame
		
	def get_properties(self):
		"""Query current camera properties via OpenCV."""
		if not self._started:
			raise RuntimeError("Camera not started. Call start() first.")

		props = {
		"width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
		"height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
		"fps": self.cap.get(cv2.CAP_PROP_FPS),
		"fourcc": "".join([chr((int(self.cap.get(cv2.CAP_PROP_FOURCC)) >> 8 * i) & 0xFF) for i in range(4)]),
		"brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
		"contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
		"backend": self.cap.getBackendName(),
		}
		for k, v in props.items():
			print(f"  {k}: {v}")
		return props
		
	def stop(self):
		""" Release the camera """
		if self._started and self.cap is not None:
			self.cap.release()
			self._started = False
			print("Thermal camera stopped")
			
	def __enter__(self):
		return self
		
	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
		
	
