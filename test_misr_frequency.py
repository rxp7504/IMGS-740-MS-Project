import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from thermal_camera import ThermalCamera

"""
Multi-Image Super Resolution using Fourier domain approach.
Captures a burst of frames, estimates sub-pixel shifts via phase correlation,
then combines frames in the frequency domain to reconstruct a higher resolution image.
"""


def estimate_shifts(frames):
    """
    Estimate sub-pixel shifts between all frames and the reference (frame 0).

    Args:
        frames: list of float32 grayscale images
    Returns:
        shifts: list of (dx, dy) tuples
    """
    reference = frames[0]
    shifts = [(0.0, 0.0)]  # reference has zero shift
    for frame in frames[1:]:
        shift, _ = cv2.phaseCorrelate(reference, frame)
        shifts.append((shift[0], shift[1]))
    return shifts


def fourier_sr(frames, shifts, scale=2):
    """
    Frequency domain super resolution.

    Args:
        frames: list of float32 grayscale LR frames
        shifts: list of (dx, dy) sub-pixel shifts per frame
        scale:  upscaling factor (default: 2)
    Returns:
        HR image as float32
    """
    h, w = frames[0].shape
    hr_h, hr_w = h * scale, w * scale

    pad_h = (hr_h - h) // 2
    pad_w = (hr_w - w) // 2

    hr_spectrum = np.zeros((hr_h, hr_w), dtype=np.complex128)

    # Precompute frequency grids
    freq_x = np.fft.fftfreq(hr_w)
    freq_y = np.fft.fftfreq(hr_h)
    Fx, Fy = np.meshgrid(freq_x, freq_y)

    for frame, (dx, dy) in zip(frames, shifts):
        # FFT of LR frame
        F = np.fft.fft2(frame)

        # Zero-pad to HR size in frequency domain (upscaling)
        F_shifted = np.fft.fftshift(F)
        F_padded  = np.pad(F_shifted, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        F_padded  = np.fft.ifftshift(F_padded)

        # Apply phase correction for sub-pixel shift
        phase_ramp = np.exp(-1j * 2 * np.pi * (Fx * dx * scale + Fy * dy * scale))
        hr_spectrum += F_padded * phase_ramp

    # Average over all frames
    hr_spectrum /= len(frames)

    # Inverse FFT to get HR image
    hr = np.real(np.fft.ifft2(hr_spectrum))

    # Scale to correct magnitude
    hr = hr * (scale ** 2)

    return hr.astype(np.float32)


def shift_and_add(frames, shifts, scale=2):
    """
    Basic shift-and-add super resolution for comparison.

    Args:
        frames: list of float32 grayscale LR frames
        shifts: list of (dx, dy) sub-pixel shifts per frame
        scale:  upscaling factor
    Returns:
        HR image as float32
    """
    h, w = frames[0].shape
    hr_h, hr_w = h * scale, w * scale

    hr_acc    = np.zeros((hr_h, hr_w), dtype=np.float32)
    hr_weight = np.zeros((hr_h, hr_w), dtype=np.float32)

    for frame, (dx, dy) in zip(frames, shifts):
        # Upsample to HR grid
        frame_up = cv2.resize(frame, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)

        # Apply shift on HR grid
        M = np.float32([[1, 0, dx * scale], [0, 1, dy * scale]])
        frame_shifted = cv2.warpAffine(frame_up, M, (hr_w, hr_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
        hr_acc    += frame_shifted
        hr_weight += 1.0

    return (hr_acc / hr_weight).astype(np.float32)


if __name__ == "__main__":

    print("=" * 70)
    print("[TEST] MULTI-IMAGE SUPER RESOLUTION")
    print("=" * 70)
    print()

    # --- Capture burst ---
    n_frames = 16
    print(f"Capturing {n_frames} frames (hold camera naturally)...")
    with ThermalCamera() as cam:
        cam.start(mode="standard")
        frames = []
        for i in range(n_frames):
            frame = cv2.rotate(cam.capture(),cv2.ROTATE_180)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
            print(f"  Frame {i+1}/{n_frames}")

    h, w = frames[0].shape
    print(f"\nBurst shape: {len(frames)} x {h} x {w}")

    # --- Estimate shifts ---
    print("\nEstimating sub-pixel shifts...")
    shifts = estimate_shifts(frames)
    for i, (dx, dy) in enumerate(shifts):
        print(f"  Frame {i:2d}: dx={dx:+.3f}  dy={dy:+.3f} px")

    shift_arr = np.array(shifts)
    print(f"\n  X range: {shift_arr[:,0].min():.3f} to {shift_arr[:,0].max():.3f} px")
    print(f"  Y range: {shift_arr[:,1].min():.3f} to {shift_arr[:,1].max():.3f} px")

    # --- Filter frames to sub-pixel shifts only ---
    scale = 2
    print("\nFiltering frames with shifts > 1px...")
    good_frames = []
    good_shifts = []
    for frame, (dx, dy) in zip(frames, shifts):
        if abs(dx) < 1.0 and abs(dy) < 1.0:
            good_frames.append(frame)
            good_shifts.append((dx, dy))

    print(f"Using {len(good_frames)}/{len(frames)} frames after filtering")

    # --- Run methods ---
    print(f"\nRunning super resolution (scale={scale}x)...")

    bicubic = cv2.resize(frames[0], (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

    t0 = time.time()
    sr_saa = shift_and_add(good_frames, good_shifts, scale=scale)
    print(f"  Shift & Add:   {(time.time()-t0)*1000:.1f} ms")

    t0 = time.time()
    sr_fft = fourier_sr(good_frames, good_shifts, scale=scale)
    print(f"  Fourier SR:    {(time.time()-t0)*1000:.1f} ms")

    # --- Run methods ---
    scale = 2
    print(f"\nRunning super resolution (scale={scale}x)...")

    # Bicubic baseline
    t0 = time.time()
    bicubic = cv2.resize(frames[0], (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    print(f"  Bicubic:       {(time.time()-t0)*1000:.1f} ms")

    # Shift and add
    t0 = time.time()
    sr_saa = shift_and_add(frames, shifts, scale=scale)
    print(f"  Shift & Add:   {(time.time()-t0)*1000:.1f} ms")

    # Fourier SR
    t0 = time.time()
    sr_fft = fourier_sr(frames, shifts, scale=scale)
    print(f"  Fourier SR:    {(time.time()-t0)*1000:.1f} ms")

    # --- Normalize for display ---
    def norm8(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    bicubic_d = norm8(bicubic)
    saa_d     = norm8(sr_saa)
    fft_d     = norm8(sr_fft)

    # --- Display ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bicubic_d, cmap='inferno')
    axes[0].set_title(f"Bicubic ({bicubic_d.shape[1]}x{bicubic_d.shape[0]})")
    axes[0].axis('off')

    axes[1].imshow(saa_d, cmap='inferno')
    axes[1].set_title(f"Shift & Add ({saa_d.shape[1]}x{saa_d.shape[0]})")
    axes[1].axis('off')

    axes[2].imshow(fft_d, cmap='inferno')
    axes[2].set_title(f"Fourier SR ({fft_d.shape[1]}x{fft_d.shape[0]})")
    axes[2].axis('off')

    plt.suptitle(f"Multi-Image SR — {n_frames} frames, {scale}x upscale", fontsize=13)
    plt.tight_layout()
    plt.savefig("_imgs/misr_result.png", dpi=150)
    plt.show()

    # --- Save results ---
    cv2.imwrite("_imgs/misr_bicubic.tiff", bicubic)
    cv2.imwrite("_imgs/misr_saa.tiff",     sr_saa)
    cv2.imwrite("_imgs/misr_fft.tiff",     sr_fft)
    print("\nResults saved to _imgs/")

    print()
    print("=" * 70)
    print("[SUCCESS] MISR Test Complete")
    print("=" * 70)
