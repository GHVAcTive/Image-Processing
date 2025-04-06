import cv2
import os
import numpy as np

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image paths using the base path
img1_path = os.path.join(base_path, 'Images', 'Arcade-Left-RANSAC-1.jpg')
img2_path = os.path.join(base_path, 'Images', 'Arcade-Right-RANSAC-1.jpg')

# Create 'Results' folder if it doesn't exist
results_folder = os.path.join(base_path, 'Results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Output path in the 'Results' folder
output_path = os.path.join(results_folder, 'stitched_result.png')

# Load images in color
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Check if images loaded correctly
if img1 is None or img2 is None:
    print("[❌] Error loading images")
    exit()

# Create SIFT detector
sift = cv2.SIFT_create(nfeatures=500)

# Detect keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match descriptors using Lowe's ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Extract matched points
pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Compute homography FROM IMG2 TO IMG1
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

if H is None:
    print("[❌] Homography calculation failed")
    exit()

# Calculate output dimensions
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# Get corners of img2
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

# Transform corners using homography
transformed_corners = cv2.perspectiveTransform(corners_img2, H)

# Combine with img1 corners to find bounds
all_corners = np.concatenate((
    transformed_corners, 
    np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
))

# Calculate panorama size
x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 1)
x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 1)
panorama_width = x_max - x_min
panorama_height = y_max - y_min

# Create translation matrix to avoid negative coordinates (with proper float type)
translation_matrix = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]], dtype=np.float32)
H_adjusted = translation_matrix.dot(H)

# Warp images
warped_img2 = cv2.warpPerspective(img2, H_adjusted, (panorama_width, panorama_height))
warped_img1 = cv2.warpPerspective(img1, translation_matrix, (panorama_width, panorama_height))

# Create result image by taking pixels from warped_img1 where available, else from warped_img2
result = np.where(warped_img1 > 0, warped_img1, warped_img2)

# Create a 2D overlap mask by summing over the color channels
mask1 = np.sum(warped_img1, axis=2) > 0
mask2 = np.sum(warped_img2, axis=2) > 0
overlap_mask = mask1 & mask2

# Blend the overlapping region using weighted addition
blended = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0)
result[overlap_mask] = blended[overlap_mask]

# Save and display the stitched result
cv2.imwrite(output_path, result)
cv2.imshow("Stitched Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[✔] Result saved: {output_path}")
