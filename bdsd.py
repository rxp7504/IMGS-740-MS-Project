import numpy as np
import cv2
from scipy.signal import kaiser
from scipy.ndimage import convolve


def mtf_pan(I_PAN, ratio, GNyq=0.15):
    """
    MTF filter for the PAN image using a Gaussian filter matched to the
    Modulation Transfer Function (MTF) of the panchromatic sensor.

    Args:
        I_PAN:  2D numpy array, panchromatic image (float64)
        ratio:  int, scale ratio between MS and PAN
        GNyq:   float, Nyquist gain of the PAN sensor (default 0.15 for generic sensor)

    Returns:
        I_Filtered: 2D numpy array, MTF-filtered PAN image
    """
    N = 41
    fcut = 1.0 / ratio

    alpha = np.sqrt((N * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))

    # Gaussian kernel (equivalent to MATLAB fspecial('gaussian', N, alpha))
    half = N // 2
    x = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, x)
    H = np.exp(-(xx ** 2 + yy ** 2) / (2 * alpha ** 2))
    H = H / H.max()

    # Kaiser window (equivalent to MATLAB kaiser(N))
    win_1d = kaiser(N, 14.0)  # beta=14 is MATLAB default
    win_2d = np.outer(win_1d, win_1d)

    # fwind1 equivalent: element-wise multiply kernel by 2D window
    h = H * win_2d
    h = h / h.sum()

    I_PAN_LP = convolve(I_PAN.astype(np.float64), np.real(h), mode='nearest')
    return I_PAN_LP


def mtf(I_MS, ratio, GNyq=None):
    """
    MTF filter for the MS image using a Gaussian filter matched to the
    Modulation Transfer Function (MTF) of the multispectral sensor.

    Args:
        I_MS:   3D numpy array (H, W, Bands), multispectral image (float64)
        ratio:  int, scale ratio between MS and PAN
        GNyq:   float or list of floats, Nyquist gain per band.
                If None, defaults to 0.29 for all bands (generic sensor).

    Returns:
        I_Filtered: 3D numpy array, MTF-filtered MS image
    """
    N = 41
    fcut = 1.0 / ratio
    nBands = I_MS.shape[2]

    if GNyq is None:
        GNyq = [0.29] * nBands
    elif isinstance(GNyq, float):
        GNyq = [GNyq] * nBands

    I_MS_LP = np.zeros_like(I_MS, dtype=np.float64)

    half = N // 2
    x = np.arange(-half, half + 1)
    xx, yy = np.meshgrid(x, x)
    win_1d = kaiser(N, 14.0)
    win_2d = np.outer(win_1d, win_1d)

    for ii in range(nBands):
        alpha = np.sqrt((N * (fcut / 2)) ** 2 / (-2 * np.log(GNyq[ii])))
        H = np.exp(-(xx ** 2 + yy ** 2) / (2 * alpha ** 2))
        H = H / H.max()
        h = H * win_2d
        h = h / h.sum()
        I_MS_LP[:, :, ii] = convolve(
            I_MS[:, :, ii].astype(np.float64), np.real(h), mode='nearest'
        )

    return I_MS_LP


def _estimate_gamma(hs_LP_d, hs_orig, pan_LP_d):
    """
    Estimate the gamma injection coefficients at reduced resolution.
    Equivalent to MATLAB estimate_gamma_cube.

    Args:
        hs_LP_d:    3D array (h, w, Nb), low-pass downsampled MS
        hs_orig:    3D array (h, w, Nb), original MS at reduced resolution
        pan_LP_d:   2D array (h, w), low-pass downsampled PAN

    Returns:
        gamma: 2D array (Nb+1, Nb), injection coefficients
    """
    h, w, Nb = hs_LP_d.shape
    n_pixels = h * w

    # Build Hd matrix: [MS bands | PAN] each flattened
    Hd = np.zeros((n_pixels, Nb + 1))
    for k in range(Nb):
        b = hs_LP_d[:, :, k]
        Hd[:, k] = b.ravel()
    Hd[:, Nb] = pan_LP_d.ravel()

    # Least squares: B = (Hd'Hd)^-1 Hd'
    B = np.linalg.pinv(Hd)

    gamma = np.zeros((Nb + 1, Nb))
    for k in range(Nb):
        b = hs_orig[:, :, k].ravel()
        bd = hs_LP_d[:, :, k].ravel()
        gamma[:, k] = B @ (b - bd)

    return gamma


def _inject_details(hs, pan, gamma):
    """
    Inject spatial details into the MS image using gamma coefficients.
    Equivalent to MATLAB compH_inject.

    Args:
        hs:     3D array (H, W, Nb), upsampled MS image
        pan:    2D array (H, W), PAN image
        gamma:  2D array (Nb+1, Nb), injection coefficients

    Returns:
        ms_en:  3D array (H, W, Nb), enhanced MS image
    """
    H, W, Nb = hs.shape
    n_pixels = H * W

    # Build H matrix: [MS bands | PAN] each flattened
    H_mat = np.zeros((n_pixels, Nb + 1))
    for k in range(Nb):
        b = hs[:, :, k]
        H_mat[:, k] = b.ravel()
    H_mat[:, Nb] = pan.ravel()

    g = gamma[:Nb + 1, :Nb]

    ms_en = np.zeros_like(hs)
    for k in range(Nb):
        b = hs[:, :, k].ravel()
        b_en = b + H_mat @ g[:, k]
        ms_en[:, :, k] = b_en.reshape(H, W)

    return ms_en


def bdsd(I_MS, I_PAN, ratio=4, S=128, GNyq_pan=0.15, GNyq_ms=None):
    """
    BDSD: Band-Dependent Spatial-Detail pansharpening algorithm.

    Fuses a low-resolution multispectral image with a high-resolution
    panchromatic image using block-wise gamma coefficient estimation.

    Args:
        I_MS:       3D numpy array (H, W, Bands), MS image upsampled to PAN scale.
                    Expected dtype: float64 or will be cast.
        I_PAN:      2D numpy array (H, W), panchromatic image.
                    Expected dtype: float64 or will be cast.
        ratio:      int, scale ratio between MS and PAN (default: 4).
                    Must be an integer. Image dims must be divisible by S,
                    and S must be divisible by ratio.
        S:          int, block size for local gamma estimation (default: 128).
                    Must be even and a multiple of ratio.
        GNyq_pan:   float, Nyquist gain for PAN MTF filter (default: 0.15).
        GNyq_ms:    float or list, Nyquist gain(s) for MS MTF filter.
                    If None, defaults to 0.29 per band.

    Returns:
        I_Fus_BDSD: 3D numpy array (H, W, Bands), pansharpened image.

    Raises:
        ValueError: If block size or ratio constraints are not met.
    """
    # --- Input validation ---
    if S > 1:
        if S % 2 != 0:
            raise ValueError("Block size S must be even.")
        if S % ratio != 0:
            raise ValueError("Block size S must be a multiple of ratio.")
        N, M = I_PAN.shape
        if N % S != 0 or M % S != 0:
            raise ValueError(
                f"PAN image dimensions ({N}x{M}) must be multiples of S={S}. "
                f"Consider cropping or resizing."
            )

    I_MS = I_MS.astype(np.float64)
    I_PAN = I_PAN.astype(np.float64)

    Nb = I_MS.shape[2]
    N, M = I_PAN.shape
    block_lr = S // ratio  # block size at reduced resolution

    # --- Reduced resolution ---
    # Low-pass filter PAN and downsample (MATLAB: pan_LP(3:ratio:end, 3:ratio:end))
    pan_LP = mtf_pan(I_PAN, ratio, GNyq=GNyq_pan)
    pan_LP_d = pan_LP[2::ratio, 2::ratio]  # 0-indexed: start at index 2

    # Downsample MS and apply MTF filter
    ms_orig = cv2.resize(
        I_MS, (M // ratio, N // ratio), interpolation=cv2.INTER_LINEAR
    )
    if ms_orig.ndim == 2:
        ms_orig = ms_orig[:, :, np.newaxis]
    ms_LP_d = mtf(ms_orig, ratio, GNyq=GNyq_ms)

    # --- Block-wise gamma estimation at reduced resolution ---
    lr_h, lr_w = ms_orig.shape[:2]
    n_blocks_r = lr_h // block_lr
    n_blocks_c = lr_w // block_lr

    # Store gamma for each block (will be tiled back to full resolution)
    gamma_map = np.zeros((N, M, Nb + 1, Nb))

    for bi in range(n_blocks_r):
        for bj in range(n_blocks_c):
            r0, r1 = bi * block_lr, (bi + 1) * block_lr
            c0, c1 = bj * block_lr, (bj + 1) * block_lr

            block_ms_LP = ms_LP_d[r0:r1, c0:c1, :]
            block_ms_orig = ms_orig[r0:r1, c0:c1, :]
            block_pan_LP = pan_LP_d[r0:r1, c0:c1]

            gamma = _estimate_gamma(block_ms_LP, block_ms_orig, block_pan_LP)

            # Tile gamma back to full-resolution block
            R0, R1 = bi * S, (bi + 1) * S
            C0, C1 = bj * S, (bj + 1) * S
            gamma_map[R0:R1, C0:C1, :, :] = gamma[np.newaxis, np.newaxis, :, :]

    # --- Fusion at full resolution ---
    I_Fus_BDSD = np.zeros_like(I_MS)

    for bi in range(n_blocks_r):
        for bj in range(n_blocks_c):
            R0, R1 = bi * S, (bi + 1) * S
            C0, C1 = bj * S, (bj + 1) * S

            block_ms = I_MS[R0:R1, C0:C1, :]
            block_pan = I_PAN[R0:R1, C0:C1]
            gamma = gamma_map[R0, C0, :, :]  # same gamma for whole block

            I_Fus_BDSD[R0:R1, C0:C1, :] = _inject_details(block_ms, block_pan, gamma)

    return I_Fus_BDSD


def prepare_images(I_thermal_raw, I_rgb, ratio=4, colormap=cv2.COLORMAP_INFERNO):
    """
    Prepare thermal and RGB images for BDSD pansharpening following the
    methodology from Raimundo et al. 2021.

    Steps:
        1. Apply false color to thermal to get 3 false color bands
        2. Build 4-band PS-MS image (false color RGB + raw thermal grayscale)
        3. Resize RGB to be exactly ratio x thermal resolution
        4. Grayscale RGB to create PAN image
        5. Upsample PS-MS to PAN resolution (nearest neighbor per paper)
        6. Crop both to be divisible by S*ratio if needed

    Args:
        I_thermal_raw:  2D numpy array (H, W), raw thermal grayscale (uint8 or float)
        I_rgb:          3D numpy array (H, W, 3), RGB image in BGR format
        ratio:          int, upscaling ratio (default: 4)
        colormap:       OpenCV colormap for false color (default: INFERNO)

    Returns:
        I_MS_up:    3D array (H*ratio, W*ratio, 4), upsampled PS-MS image
        I_PAN:      2D array (H*ratio, W*ratio), grayscale PAN image
        th_h, th_w: original thermal dimensions (for reference)
    """
    # Thermal dimensions
    th_h, th_w = I_thermal_raw.shape[:2]
    pan_h, pan_w = th_h * ratio, th_w * ratio

    # Step 1: False color thermal (3 bands)
    if I_thermal_raw.dtype != np.uint8:
        thermal_8bit = cv2.normalize(
            I_thermal_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
    else:
        thermal_8bit = I_thermal_raw

    false_color = cv2.applyColorMap(thermal_8bit, colormap)  # BGR, uint8

    # Step 2: Build 4-band PS-MS at thermal resolution
    # Bands: [B_fc, G_fc, R_fc, thermal_gray]
    ps_ms = np.zeros((th_h, th_w, 4), dtype=np.float64)
    ps_ms[:, :, 0] = false_color[:, :, 0].astype(np.float64)  # B
    ps_ms[:, :, 1] = false_color[:, :, 1].astype(np.float64)  # G
    ps_ms[:, :, 2] = false_color[:, :, 2].astype(np.float64)  # R
    ps_ms[:, :, 3] = thermal_8bit.astype(np.float64)           # raw thermal gray

    # Step 3: Resize RGB to pan_h x pan_w
    rgb_resized = cv2.resize(I_rgb, (pan_w, pan_h), interpolation=cv2.INTER_LINEAR)

    # Step 4: Grayscale RGB → PAN
    I_PAN = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Step 5: Upsample PS-MS to PAN resolution (nearest neighbor per paper)
    I_MS_up = cv2.resize(
        ps_ms, (pan_w, pan_h), interpolation=cv2.INTER_NEAREST
    )

    return I_MS_up, I_PAN, th_h, th_w


def crop_to_block_multiple(I_MS, I_PAN, S=128):
    """
    Crop images so their dimensions are multiples of S.
    Required by BDSD block processing.

    Args:
        I_MS:   3D array (H, W, Bands)
        I_PAN:  2D array (H, W)
        S:      int, block size

    Returns:
        I_MS_cropped, I_PAN_cropped
    """
    H, W = I_PAN.shape
    H_crop = (H // S) * S
    W_crop = (W // S) * S
    return I_MS[:H_crop, :W_crop, :], I_PAN[:H_crop, :W_crop]
