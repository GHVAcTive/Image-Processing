import cv2
import numpy as np
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image path using the base path
img_path = os.path.join(base_path, 'Images', 'Geometric_Art-Harris_Corner.jpg')  # Replace with your image filename

# Load image in color
img = cv2.imread(img_path)

# Check if image loaded correctly
if img is None:
    print("[❌] Error loading image")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32
gray_float32 = np.float32(gray)

# Harris corner detection
dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)

# Dilate the corner image with a larger kernel to make the corners bigger
dst = cv2.dilate(dst, None, iterations=5)  # Increase iterations for more dilation

# Increase the corner size further by applying a lower threshold to make more corners visible
threshold = 0.01 * dst.max()  # Reduce the threshold for a broader selection
img[dst > threshold] = [255, 0, 0]  # Color the corners in green (BGR: [0, 255, 0])

# Display the image with corners marked
cv2.imshow('Harris Corner Detection', img)

# Save the result
output_path = os.path.join(base_path, 'Results', 'harris_corners_result.png')
cv2.imwrite(output_path, img)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[✔] Result saved: {output_path}")
