# 🧠 Computer Vision Project Report: Feature Detection & Matching

---

## 📋 Table of Contents

1. [📌 Introduction](#introduction)
2. [📁 Project Structure](#project-structure)
3. [🛠 Prerequisites](#prerequisites)
4. [📦 Installation & Setup](#installation--setup)
5. [🧪 Algorithms Implemented](#algorithms-implemented)  
   - [🔹 Harris Corner Detection](#harris-corner-detection)  
   - [🔹 Shi-Tomasi Corner Detection](#shi-tomasi-corner-detection)  
   - [🔹 SIFT (Scale-Invariant Feature Transform)](#sift)  
   - [🔹 SURF (Speeded-Up Robust Features)](#surf)  
   - [🔹 ORB (Oriented FAST and Rotated BRIEF)](#orb)  
   - [🔹 RANSAC (Random Sample Consensus)](#ransac)  
6. [📊 Experimental Results](#experimental-results)
7. [⚠️ Troubleshooting Tips](#troubleshooting-tips)
8. [📚 References](#references)

---

## 📌 Introduction

In this project, we explore a suite of widely used feature detection and matching algorithms in computer vision. The primary goal is to extract and visualize important points or features from input images. These features are crucial in applications such as object tracking, panorama stitching, motion analysis, and more.

Each algorithm brings its own benefits depending on image scale, rotation, and illumination. This report offers an intuitive understanding, implementation details, and a visual demonstration of each technique.

---

## 📁 Project Structure

```
Image-Processing/
│
├── Algorithm/
│   ├── Harris_Corner/
│   ├── Shi-Tomasi/
│   ├── SIFT/
│   ├── SURF/
│   ├── ORB/
│   └── RANSAC/
│
├── Images/
│   └── <All Input Images>
│
├── Results/
│   └── <Processed Output Screenshots>
│
└── README.md
```

---

## 🛠 Prerequisites

Make sure the following libraries are installed:

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 📦 Installation & Setup

1. Clone the Repository:
```bash
git clone <repository_url>
cd Image-Processing
```

2. Place your input images inside the `Images/` folder.

3. Navigate to the respective algorithm folder and run the scripts.

---

## 🧪 Algorithms Implemented

---

### 🔹 Harris Corner Detection

#### 📖 Theory & Background

Harris Corner Detection is one of the oldest and most reliable methods to identify interest points in an image. It detects points where the gradient of intensity changes significantly in multiple directions.

- Proposed by Chris Harris and Mike Stephens in 1988.
- Computes a **corner response function** based on eigenvalues of the structure tensor.
- Sensitive to scale and rotation.

#### 📂 Implementation Details

- Image is converted to grayscale.
- `cv2.cornerHarris()` is used to compute the corner response.
- Dilation and thresholding applied to mark strong corners in **blue**.

#### 📄 Code Snippet

>  Here’s the code snippet:

```bash
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
```

#### 🖼 Output

> Here is the Output: 
![harris_corners_result](https://github.com/user-attachments/assets/658c2a42-99fb-4f29-9a2b-fe47b7807755)


---

### 🔹 Shi-Tomasi Corner Detection

#### 📖 Theory & Background

Shi-Tomasi, or "Good Features to Track", improves Harris by selecting corners based on the **minimum eigenvalue** rather than the sum.

- Proposed by Jianbo Shi and Carlo Tomasi (1994).
- More robust for motion tracking and optical flow.

#### 📂 Implementation Details

- `cv2.goodFeaturesToTrack()` identifies the best `N` corners.
- Large **green circles** used for visibility.
- Adjustable parameters: `maxCorners`, `qualityLevel`, `minDistance`.

#### 📄 Code Snippet

> Here is the Code :
```bash
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
```

#### 🖼 Output

> 
![shi_tomasi_corners_large_result](https://github.com/user-attachments/assets/c7824cd0-3369-4428-a6cb-66de56b9b818)

---

### 🔹 SIFT (Scale-Invariant Feature Transform)

#### 📖 Theory & Background

SIFT is a powerful keypoint detection and descriptor extraction technique.

- Introduced by David Lowe (1999).
- Detects features invariant to **scale**, **rotation**, and **illumination**.
- Builds a scale-space and uses DOG (Difference of Gaussian) for keypoint detection.

#### 📂 Implementation Details

- Uses `cv2.SIFT_create()` to detect and compute descriptors.
- Features are matched using Brute Force Matcher (`BFMatcher`) and drawn using `cv2.drawMatches()`.

#### 📄 Code Snippet

> _[Insertedd SIFT Code here]_

```bash
import cv2
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image paths using the base path
img1_path = os.path.join(base_path, 'Images', 'BOX-SIFT-1.jpg')
img2_path = os.path.join(base_path, 'Images', 'BOX-SIFT-2.jpg')

# Create 'Results' folder if it doesn't exist
results_folder = os.path.join(base_path, 'Results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Output path in the 'Results' folder
output_path = os.path.join(results_folder, 'sift_result.jpg')

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check if images loaded correctly
if img1 is None or img2 is None:
    print("[❌] Error loading one or both images. Check file paths.")
    exit()

# Resize img2 to match img1's dimensions
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Create SIFT detector
sift = cv2.SIFT_create()

# Detect and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2_resized, None)

# Match descriptors using Brute-Force and apply Lowe's ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Draw good matches with custom line color and thickness
result_img = cv2.drawMatches(
    img1, kp1, img2_resized, kp2, good_matches, None, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # This flag disables single point drawing
)

# Save result
cv2.imwrite(output_path, result_img)
print(f"[✔] SIFT result saved at: {output_path}")
```

#### 🖼 Output

> _[Inserted SIFT Output Screenshot here]_

![sift_result](https://github.com/user-attachments/assets/cb6573bd-5b77-449b-ae8a-c1e85c730855)


---

### 🔹 SURF (Speeded-Up Robust Features)

#### 📖 Theory & Background

SURF is an optimized version of SIFT designed to be faster and more efficient.

- Uses an integral image for quick convolution.
- Uses **Hessian matrix** to detect blobs.

> 🔒 Note: SURF is patented and may require `opencv-contrib-python`.

#### 📂 Implementation Details

- Features are detected using `cv2.xfeatures2d.SURF_create()`.
- Descriptors matched using Brute Force.
- Works well under scale and rotation.

#### 📄 Code Snippet

> _[Inserted SURF Code here]_

```bash
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
output_path = os.path.join(results_folder, 'surf_result.png')

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check if images loaded correctly
if img1 is None or img2 is None:
    print("[❌] Error loading one or both images. Check file paths.")
    exit()

# Resize or rotate img2 to match img1's dimensions (simulate scale and orientation changes)
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # Resize img2 to match img1's dimensions

# Create SURF detector
surf = cv2.xfeatures2d.SURF_create(400)

# Detect and compute descriptors
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2_resized, None)  # Use img2_resized or img2_rotated depending on your choice

# Match descriptors using Brute-Force and apply Lowe's ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Draw good matches with custom line color and thickness
result_img = cv2.drawMatches(
    img1, kp1, img2_resized, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  # This flag disables single point drawing
)

# Save result
cv2.imwrite(output_path, result_img)
print(f"[✔] SURF result saved at: {output_path}")
```

#### 🖼 Output

> _[Inserted SURF Output Screenshot here]_

![orb_result](https://github.com/user-attachments/assets/c61c22aa-a394-489e-bcdb-d8059fdc2702)


---

### 🔹 ORB (Oriented FAST and Rotated BRIEF)

#### 📖 Theory & Background

ORB is an open-source alternative to SIFT and SURF.

- Combines FAST keypoint detector with BRIEF descriptors.
- Rotation and scale invariant.
- Very efficient and suitable for real-time apps.

#### 📂 Implementation Details

- Uses `cv2.ORB_create()` to extract features.
- Binary descriptors matched using `BFMatcher` with Hamming norm.

#### 📄 Code Snippet

> _[Inserted ORB Code here]_

```bash
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
```

#### 🖼 Output

> _[Inserted ORB Output Screenshot here]_
> Same as SURF

---

### 🔹 RANSAC (Random Sample Consensus)

#### 📖 Theory & Background

RANSAC is a robust method to estimate a model from noisy data.

- Iteratively selects random sample sets and computes a transformation matrix (e.g. Homography).
- Outliers are rejected to improve match accuracy.

#### 📂 Implementation Details

- Feature points detected using SIFT/ORB.
- Good matches filtered using `cv2.findHomography(..., cv2.RANSAC)`.
- Final result is a warped image or matched keypoints excluding outliers.

#### 📄 Code Snippet

> _[Inserted RANSAC Code here]_

```bash
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
```

#### 🖼 Output

> _[Inserted RANSAC Output Screenshot here]_

![stitched_result](https://github.com/user-attachments/assets/212a4f8b-e731-4a98-8c12-c1667237b90b)

---

## 📊 Experimental Results

| Algorithm       | Invariance       | Robustness | Execution Time | Best Use Case                      |
|----------------|------------------|------------|----------------|-------------------------------------|
| Harris          | Rotation         | Medium     | Fast           | Simple corner detection             |
| Shi-Tomasi      | Rotation         | High       | Fast           | Optical flow, tracking              |
| SIFT            | Scale + Rotation | High       | Medium         | Matching & Recognition              |
| SURF            | Scale + Rotation | High       | Faster than SIFT | Feature-rich environments        |
| ORB             | Scale + Rotation | Medium     | Very Fast      | Mobile & Real-Time Applications     |
| RANSAC          | Model Fitting    | High       | Varies         | Panorama Stitching, Outlier Removal |

---

## ⚠️ Troubleshooting Tips

| Issue                       | Solution                                                       |
|----------------------------|----------------------------------------------------------------|
| Output window not showing  | Use local GUI environment (avoid headless/WSL)                 |
| No keypoints detected      | Lower qualityLevel or threshold, check image resolution        |
| Errors in SURF             | Ensure you're using `opencv-contrib-python`                   |
| RANSAC mismatches          | Tune RANSAC threshold or use better descriptor matches         |

---

## 📚 References

1. Harris & Stephens, 1988 – “A Combined Corner and Edge Detector”
2. Shi & Tomasi, 1994 – “Good Features to Track”
3. Lowe, D.G., 2004 – “Distinctive Image Features from Scale-Invariant Keypoints”
4. Bay et al., 2008 – “Speeded-Up Robust Features (SURF)”
5. Fischler & Bolles, 1981 – “Random Sample Consensus”

---

## 📦 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

Let me know if you want this as a downloadable `.docx` or `.md` file, or if you'd like the content split across separate reports per algorithm!
