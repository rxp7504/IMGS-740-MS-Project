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
		self._mode = None
		
	def start(self,mode="standard"):
		""" 
		Open and configure the thermal camera.
		
		Args:
			mode: "standard" capture a UYVY for standard BGR frame
				  "radiometry" capture Y16 raw data for real temperature values
		"""
		self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
		
		if mode == "standard":
			self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'UYVY'))
		elif mode == "radiometry":
			self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
			self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) # don't auto-convert
		else:
			raise RuntimeError("Invalid camera mode. Please select 'standard' or 'radiometry'")
			
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
		self.cap.set(cv2.CAP_PROP_FPS, 9)
		
		self._mode = mode
		
		if not self.cap.isOpened():
			raise RuntimeError(f"Could not open thermal camera at /dev/video{self.device_index}")
			
		# flush the buffer
		for _ in range(5):
			self.cap.grab()
			
		self._started = True
		print(f"Thermal camera started on /dev/video{self.device_index} in {mode} mode")
			
	def capture(self):
		"""
		Capture a raw thermal frame.
		
		Returns:
			In standard mode: numpy array (H, W, 3) in BGR format
			In radiometry mode: numpy array (H, W) uint16 raw sensor counts
		"""
		if not self._started:
			raise RuntimeError("Camera not started. Call start() first.")
		
		# Flush stale buffered frames before capturing
		for _ in range(5):
			self.cap.grab()	
		
		ret, frame = self.cap.read()
		
		if not ret:
			raise RuntimeError("Failed to capture thermal frame")
			
		return frame
		
	def convert_raw(self,raw,unit="c"):
		"""
		Convert a raw Y16 capture to temperature units.
		
		Args:
			raw: raw thermal capture in uint16 format
			unit: the desired units to convert to
				  "k" Kelvin
				  "c" Celsius
				  "f" Fahrenheit
		Returns:
			numpy array (H, W) float32 in the desired units
		"""
		raw = raw.astype(np.float32)
		
		if unit == "k":
			frame_out = (raw / 100.0)
		elif unit == "c":
			frame_out = (raw / 100.0) - 273.15
		elif unit == "f":
			frame_out = ((raw / 100.0) - 273.15) * (9/5) + 32
		else:
			raise ValueError("Invalid temperature unit. Please select 'k','c',or 'f'")
			
		return frame_out
		
		
		
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
		
	
