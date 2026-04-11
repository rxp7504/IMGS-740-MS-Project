import cv2
import numpy as np
import matplotlib.pyplot as plt
from thermal_camera import ThermalCamera

def shift_and_add(frames, scale=2):
    h, w = frames[0].shape
    hr_h, hr_w = h * scale, w * scale

    hr_acc    = np.zeros((hr_h, hr_w), dtype=np.float32)
    hr_weight = np.zeros((hr_h, hr_w), dtype=np.float32)

    reference = frames[0]

    for frame in frames:
        shift, _ = cv2.phaseCorrelate(reference, frame)
        dx, dy = shift

        # For each LR pixel, find its position on the HR grid
        for y in range(h):
            for x in range(w):
                # HR position of this LR pixel
                hr_x = x * scale + dx * scale
                hr_y = y * scale + dy * scale

                # Distribute value to surrounding HR pixels (bilinear splat)
                x0, y0 = int(hr_x), int(hr_y)
                fx, fy = hr_x - x0, hr_y - y0

                for ix, wx in [(x0, 1-fx), (x0+1, fx)]:
                    for iy, wy in [(y0, 1-fy), (y0+1, fy)]:
                        if 0 <= ix < hr_w and 0 <= iy < hr_h:
                            w_val = wx * wy
                            hr_acc[iy, ix]    += frame[y, x] * w_val
                            hr_weight[iy, ix] += w_val

    # Normalize
    mask = hr_weight > 0
    hr = np.zeros_like(hr_acc)
    hr[mask] = hr_acc[mask] / hr_weight[mask]

    # Fill any unfilled pixels with bicubic interpolation
    if not mask.all():
        hr_bicubic = cv2.resize(reference, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        hr[~mask] = hr_bicubic[~mask]

    return hr


if __name__ == "__main__":

    # Capture burst
    with ThermalCamera() as cam:
        cam.start(mode="standard")
        frames = []
        for i in range(16):
            frame = cam.capture()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
            print(f"Captured frame {i+1}/16")

    # Use only first 8 frames (before drift exceeds 1px)
    frames = frames[:16]

    # Bicubic baseline for comparison
    h, w = frames[0].shape
    bicubic = cv2.resize(frames[0], (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # Shift and add
    sr = shift_and_add(frames, scale=2)

    # Normalize both for display
    bicubic_norm = cv2.normalize(bicubic, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    sr_norm      = cv2.normalize(sr,      None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(bicubic_norm, cmap='inferno')
    ax[0].set_title(f"Bicubic {bicubic_norm.shape}")
    ax[0].axis('off')
    ax[1].imshow(sr_norm, cmap='inferno')
    ax[1].set_title(f"Shift and Add {sr_norm.shape}")
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig("_imgs/sr_result.png")
    plt.show()

    print("Done!")
