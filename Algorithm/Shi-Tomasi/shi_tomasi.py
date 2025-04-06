import cv2
import numpy as np
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image path using the base path
img_path = os.path.join(base_path, 'Images', 'Giraffe-Shi_Tomasi-2.jpg')  # Replace with your image filename

# Load image in color
img = cv2.imread(img_path)

# Check if image loaded correctly
if img is None:
    print("[❌] Error loading image")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Shi-Tomasi corners
corners = cv2.goodFeaturesToTrack(gray, 
                                  maxCorners=100,  # maximum number of corners to return
                                  qualityLevel=0.01,  # quality level
                                  minDistance=10,  # minimum distance between corners
                                  blockSize=3)  # size of the neighborhood used for corner detection

# Convert corners to integer values for drawing
corners = np.int32(corners)  # Use np.int32 for proper integer conversion

# Draw very large circles on the original image (increased size for visibility)
for corner in corners:
    x, y = corner.ravel()  # Unroll the corner coordinates
    cv2.circle(img, (x, y), 80, (0, 255, 0), 10)  # Very large green circles (80 radius, thickness 10)

# Display the image with corners marked
cv2.imshow('Shi-Tomasi Corners - Large', img)

# Save the result
output_path = os.path.join(base_path, 'Results', 'shi_tomasi_corners_large_result.png')
cv2.imwrite(output_path, img)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"[✔] Result saved: {output_path}")
