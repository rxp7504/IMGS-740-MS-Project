from flask import Flask, render_template, redirect, url_for, jsonify
import cv2
import numpy as np
import os
import time
import threading
from pi_camera import RGBCamera
from thermal_camera import ThermalCamera
import utils
import tifffile

app = Flask(__name__)

CAPTURE_DIR = "static/captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

last_capture   = {"rgb": None, "thermal": None, "ts": None}
last_sharpened = {"result": None, "status": "idle", "error": None}

# Calibration resources
K     = np.load("_resources/thermal_K.npy")
dist  = np.load("_resources/thermal_dist.npy")
H     = np.load("_resources/H_20x.npy")
RATIO = 20

# Initialize cameras
rgb_cam = RGBCamera()
rgb_cam.start(mode="still")
thermal_cam = ThermalCamera()
thermal_cam.start(mode="radiometry")
print("Cameras Initialized")


@app.route("/")
def index():
    return render_template("index.html",
                           images=last_capture,
                           sharpened=last_sharpened)


@app.route("/capture", methods=["POST"])
def capture():
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # RGB Camera
    rgb_frame    = rgb_cam.capture()
    rgb_filename = f"rgb_{timestamp}.jpg"
    cv2.imwrite(os.path.join(CAPTURE_DIR, rgb_filename), rgb_frame)

    # Thermal Camera
    thermal_frame = thermal_cam.capture()
    thermal_frame = cv2.rotate(thermal_frame, cv2.ROTATE_180)
    temp = (cv2.normalize(thermal_frame.astype(np.float32), None, 0.0, 1.0,
                          cv2.NORM_MINMAX) * 255).astype(np.uint8)
    thermal_8bit          = cv2.applyColorMap(temp, cv2.COLORMAP_INFERNO)
    thermal_filename      = f"thermal_{timestamp}.tiff"
    thermal_filename_8bit = f"thermal_{timestamp}_8bit.jpg"
    cv2.imwrite(os.path.join(CAPTURE_DIR, thermal_filename), thermal_frame)
    cv2.imwrite(os.path.join(CAPTURE_DIR, thermal_filename_8bit), thermal_8bit)
    print(f"Saved: {rgb_filename}, {thermal_filename}")

    last_capture["rgb"]     = rgb_filename
    last_capture["thermal"] = thermal_filename_8bit
    last_capture["ts"]      = timestamp

    # Reset sharpened state on new capture
    last_sharpened["result"] = None
    last_sharpened["status"] = "idle"
    last_sharpened["error"]  = None

    return redirect(url_for("index"))


@app.route("/sharpen", methods=["POST"])
def sharpen():
    if not last_capture.get("ts"):
        return redirect(url_for("index"))

    ts = last_capture["ts"]

    def run_sharpen():
        try:
            last_sharpened["status"] = "running"
            last_sharpened["error"]  = None

            thermal_path = os.path.join(CAPTURE_DIR, f"thermal_{ts}.tiff")
            rgb_path     = os.path.join(CAPTURE_DIR, f"rgb_{ts}.jpg")

            thermal_raw = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            rgb         = cv2.imread(rgb_path).astype(np.float32) / 255.0

            thermal_sharpened, pan, thermal_sharpened_raw, thermal_upscale = \
                utils.gradient_sharpen_thermal(thermal_raw, rgb, RATIO, K, dist, H)

            # Save displayable PNG
            sharpened_png = f"sharpened_{ts}.png"
            utils.save_gray(os.path.join(CAPTURE_DIR, sharpened_png), thermal_sharpened)

            # Save raw tiff for downstream use
            tifffile.imwrite(os.path.join(CAPTURE_DIR, f"sharpened_{ts}.tiff"),
                             thermal_sharpened_raw)

            last_sharpened["result"] = sharpened_png
            last_sharpened["status"] = "done"
            print(f"Sharpening complete: {sharpened_png}")

        except Exception as e:
            last_sharpened["status"] = "error"
            last_sharpened["error"]  = str(e)
            print(f"Sharpening error: {e}")

    threading.Thread(target=run_sharpen, daemon=True).start()
    return redirect(url_for("index"))


@app.route("/sharpen_status")
def sharpen_status():
    """Poll endpoint — returns JSON status for the frontend to check."""
    return jsonify(last_sharpened)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        rgb_cam.stop()
        thermal_cam.stop()
