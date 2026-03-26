import cv2
import numpy as np
import matplotlib.pyplot as plt

def inner_bounding_box(aligned):
    """ 
    Find the largest inner bounding box of a warped image with no black borders.
    
    Args:
        aligned: 2D float32 warped image
    Returns:
        x, y, w, h: bounding box coordinates
    """
    # Mask of non-black pixels
    mask = (aligned > 0).astype(np.uint8)
    
    # Find the bounding box of the non-black region
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    
    return x, y, w, h

if __name__ == "__main__":
    print("="*70)
    print("[TEST] APPLY REGISTRATION MATRIX")
    print("="*70)
    
    # Load images
    rgb = cv2.imread("_imgs/rgb_test.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.imread("_imgs/thermal_img.jpg", 0).astype(np.float32) / 255.0
    thermal = cv2.rotate(thermal, cv2.ROTATE_180)  # thermal camera is upside down

    # Load registration matrix
    warp_matrix = np.load("_resources/warp_matrix.npy")

    # Normalize
    cv2.normalize(rgb, rgb, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(thermal, thermal, 0, 1, cv2.NORM_MINMAX)

    # Resize RGB to be 4x the thermal image
    rgb = cv2.resize(rgb, (thermal.shape[1] * 4, thermal.shape[0] * 4))
    print(f"    RGB image resized to {rgb.shape}")

    # Resize thermal to match the down-res RGB
    thermal = cv2.resize(thermal, (thermal.shape[1] * 4, thermal.shape[0] * 4))
    print(f"    Thermal image resized to {thermal.shape}")
    
    # Apply to thermal
    aligned = cv2.warpAffine(
        thermal,
        warp_matrix,
        (rgb.shape[1], rgb.shape[0]),
        flags=cv2.INTER_LINEAR
    )

    # Find inner bounding box and crop both images
    x, y, w, h = inner_bounding_box(aligned)
    print(f"    Inner bounding box: x={x}, y={y}, w={w}, h={h}")
    
    rgb_cropped = rgb[y:y+h,x:x+w]
    thermal_cropped = aligned[y:y+h,x:x+w]
    print(f"    Cropped shape: {rgb_cropped.shape}")

    # Display result
    sbs2 = np.hstack((rgb_cropped, thermal_cropped))
    plt.imshow(sbs2, cmap='gray')
    plt.axis("off")
    plt.title("After Registration and Cropping")
    plt.show()
    
    print("\n[SUCCESS] Registration Test Complete")
