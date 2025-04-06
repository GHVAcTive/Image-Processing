import cv2
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image paths using the base path
img1_path = os.path.join(base_path, 'Images', 'RR-SURF-1.png')  # Front view of the car
img2_path = os.path.join(base_path, 'Images', 'RR-SURF-2.webp')  # Side view of the car (modified)

# Create 'Results' folder if it doesn't exist
results_folder = os.path.join(base_path, 'Results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Output path in the 'Results' folder
output_path = os.path.join(results_folder, 'orb_result.png')

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check if images loaded correctly
if img1 is None or img2 is None:
    print("[❌] Error loading one or both images. Check file paths.")
    exit()

# Resize or rotate img2 to match img1's dimensions (simulate scale and orientation changes)
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # Resize img2 to match img1's dimensions

# Create ORB detector
orb = cv2.ORB_create(400)

# Detect and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2_resized, None)

# Match descriptors using Brute-Force and apply Lowe's ratio test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort them in ascending order of distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw good matches with custom line color and thickness
result_img = cv2.drawMatches(
    img1, kp1, img2_resized, kp2, matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # This flag disables single point drawing
)

# Save result
cv2.imwrite(output_path, result_img)
print(f"[✔] ORB result saved at: {output_path}")
