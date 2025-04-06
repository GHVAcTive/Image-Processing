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

>  Here’s the code snippet formatted properly in **bash code block** as you requested:

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

This will allow you to include the code in your report, leaving space for screenshots of the output and making it easy to follow. Let me know if you need further modifications!


#### 🖼 Output

> _[Insert Output Screenshot here]_  

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

> _[Insert Shi-Tomasi code here]_

#### 🖼 Output

> _[Insert Output Screenshot here]_  

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

> _[Insert SIFT Code here]_

#### 🖼 Output

> _[Insert SIFT Output Screenshot here]_

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

> _[Insert SURF Code here]_

#### 🖼 Output

> _[Insert SURF Output Screenshot here]_

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

> _[Insert ORB Code here]_

#### 🖼 Output

> _[Insert ORB Output Screenshot here]_

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

> _[Insert RANSAC Code here]_

#### 🖼 Output

> _[Insert RANSAC Output Screenshot here]_

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
