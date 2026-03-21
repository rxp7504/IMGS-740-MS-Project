import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("_imgs/rgb_test.jpg", 0)   # reference
img2 = cv2.imread("_imgs/thermal_img.jpg", 0)  # to align
img2 = cv2.rotate(img2,cv2.ROTATE_180)

# Convert to float
img1 = img1.astype(np.float32) / 255.0
img2 = img2.astype(np.float32) / 255.0

# Normalize
cv2.normalize(img1,img1,0,1,cv2.NORM_MINMAX)
cv2.normalize(img2,img2,0,1,cv2.NORM_MINMAX)

# Resize thermal to match RGB
img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))

# Display images
sbs = np.hstack((img1,img2))
plt.imshow(cv2.cvtColor(sbs,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Choose motion model
warp_mode = cv2.MOTION_AFFINE
warp_matrix = np.eye(2, 3, dtype=np.float32)

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

cc, warp_matrix = cv2.findTransformECC(
    img1, img2, warp_matrix, warp_mode, criteria
)

aligned = cv2.warpAffine(
    img2,
    warp_matrix,
    (img1.shape[1], img1.shape[0]),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
)
