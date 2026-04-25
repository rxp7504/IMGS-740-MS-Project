"""
Sanity check: round-trip Poisson reconstruction
Reconstruct thermal from its own gradients and verify error is near zero.
"""

import numpy as np
import tifffile


def dst_cols(a: np.ndarray) -> np.ndarray:
    """DST applied column by column to avoid OOM — matches MATLAB dst()."""
    n = a.shape[0]
    out = np.zeros_like(a, dtype=np.float64)
    pad = np.zeros(2*(n+1), dtype=np.float64)
    for j in range(a.shape[1]):
        pad[:] = 0
        pad[1:n+1] = a[:, j]
        pad[n+2:] = -a[::-1, j]
        yy = np.fft.rfft(pad)
        out[:, j] = (yy[1:n+1] / (-2j)).real
    return out


def idst_cols(a: np.ndarray) -> np.ndarray:
    """Inverse DST column by column — matches MATLAB idst()."""
    n = a.shape[0]
    return (2.0 / (n+1)) * dst_cols(a)


def dst2(A: np.ndarray) -> np.ndarray:
    """2D DST — matches MATLAB: dst(dst(A)')' """
    return dst_cols(dst_cols(A).T).T


def idst2(A: np.ndarray) -> np.ndarray:
    """2D inverse DST — matches MATLAB: idst(idst(A)')' """
    return idst_cols(idst_cols(A).T).T

from scipy.fft import dctn, idctn

def poisson_reconstruct(gx: np.ndarray, gy: np.ndarray,
                        mean_val: float) -> np.ndarray:
    h, w = gx.shape

    div = np.zeros((h, w), dtype=np.float64)
    div[:, :-1] += np.diff(gx, axis=1)
    div[:-1, :] += np.diff(gy, axis=0)

    div_dct = dctn(div, type=2, norm='ortho')

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    eigenvalues = (2*np.cos(np.pi*x/w) - 2) + \
                  (2*np.cos(np.pi*y/h) - 2)
    eigenvalues[0, 0] = 1  # avoid division by zero

    solution = div_dct / eigenvalues
    solution[0, 0] = 0  # zero DC — correct mean separately after

    I = idctn(solution, type=2, norm='ortho')

    # Correct mean explicitly
    I = I - I.mean() + mean_val

    return I.astype(np.float32)

if __name__ == "__main__":
    print("=" * 70)
    print("[TEST] POISSON ROUND-TRIP RECONSTRUCTION")
    print("=" * 70)

    # Load thermal
    PS_MS   = tifffile.imread("_imgs/PS_MS_HR_p.tiff")
    thermal = PS_MS[:, :, -1].astype(np.float64) if PS_MS.ndim == 3 \
              else PS_MS.astype(np.float64)

    print(f"Thermal shape: {thermal.shape}")
    print(f"Thermal range: {thermal.min():.4f} to {thermal.max():.4f}")

    # Round trip
    print("\nComputing gradients...")
    gy, gx = np.gradient(thermal)

    print("Running Poisson reconstruction...")
    I_reconstructed = poisson_reconstruct(gx, gy, float(thermal.mean()))

    # Error
    err = np.abs(thermal - I_reconstructed.astype(np.float64))
    print(f"\nRound-trip error:")
    print(f"  Max error:  {err.max():.8f}")
    print(f"  Mean error: {err.mean():.8f}")
    print(f"  RMS error:  {np.sqrt(np.mean(err**2)):.8f}")
    print(f"\nScale check:")
    print(f"  Thermal mean:       {thermal.mean():.6f}")
    print(f"  Reconstructed mean: {float(I_reconstructed.mean()):.6f}")
    print(f"  Thermal std:        {thermal.std():.6f}")
    print(f"  Reconstructed std:  {float(I_reconstructed.std()):.6f}")

    if err.max() < 0.01:
        print("\n  PASS — reconstruction is accurate")
    else:
        print("\n  FAIL — check DST normalization")

    # Save error map only — avoid OOM from showing large images
    print("\nSaving error map...")
    import cv2
    err_vis = cv2.normalize(err, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    err_color = cv2.applyColorMap(err_vis, cv2.COLORMAP_HOT)
    cv2.imwrite("_imgs/roundtrip_error.png", err_color)
    print("Saved to _imgs/roundtrip_error.png")

    print("=" * 70)
    print("[DONE]")
    print("=" * 70)
