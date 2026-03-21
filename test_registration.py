import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_edges(img,blur=5,low=50,high=150):
    """ Extract edges from a greyscale image. """
    blurred = cv2.GaussianBlur(img, (blur,blur),0)
    edges = cv2.Canny((blurred * 255).astype(np.uint8),low,high)
    return edges.astype(np.float32) / 255.0

if __name__ == "__main__":

    print("="*70)
    print("[TEST] RGB AND THERMAL IMAGE REGISTRATION")
    print("="*70)
    print("\n")

    # Load images
    rgb = cv2.imread("_imgs/rgb_test.jpg", 0).astype(np.float32) / 255.0   # reference
    thermal = cv2.imread("_imgs/thermal_img.jpg", 0).astype(np.float32) / 255.0  # to align
    thermal = cv2.rotate(thermal,cv2.ROTATE_180) # thermal camera is upside down

    # Normalize
    cv2.normalize(rgb,rgb,0,1,cv2.NORM_MINMAX)
    cv2.normalize(thermal,thermal,0,1,cv2.NORM_MINMAX)

    # Resize RGB to be 4x the thermal image
    rgb = cv2.resize(rgb, (thermal.shape[1]*4,thermal.shape[0]*4))
    print(f"    RGB image resized to {rgb.shape}")

    # Resize thermal to match the down-res RGB
    thermal = cv2.resize(thermal, (thermal.shape[1]*4,thermal.shape[0]*4))
    print(f"    Thermal image resized to {thermal.shape}")
    
    # Extract edges from both
    edges_rgb = extract_edges(rgb,blur=9)
    edges_thermal = extract_edges(thermal,blur=1)

    # Display pre-registered images
    sbs = np.vstack((np.hstack((rgb,thermal)) , np.hstack((edges_rgb,edges_thermal))))
    plt.imshow(cv2.cvtColor(sbs,cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Pre-registered Images")
    plt.show()
"""
    # Choose motion model
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    print("Finding best transform...")
    cc, warp_matrix = cv2.findTransformECC(
        img1, img2, warp_matrix, warp_mode, criteria
    )
    print("Transform found!")
    aligned = cv2.warpAffine(
        img2,
        warp_matrix,
        (img1.shape[1], img1.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    # Display the registered image
    plt.imshow(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Registered Image")
    plt.show()
"""
print("[SUCCESS] Image alignment complete")
