# SIFT Algorithm - Keypoint Detection and Matching

## 📁 Project Structure

```
Image-Processing/
│
├── Algorithm/
│   └── SIFT/
│       ├── sift_match.py
│       ├── README.md
│       └── Images/
│           ├── BOX-SIFT-1.jpg
│           └── BOX-SIFT-2.jpg
└── Results/
    └── sift_result.jpg
```

---

## 📝 Overview

This project implements the **SIFT (Scale-Invariant Feature Transform)** algorithm to detect and match keypoints between two images. It uses the **OpenCV** library to perform keypoint detection, descriptor extraction, and matching, providing a simple implementation of SIFT keypoint matching. SIFT is robust to changes in scale, rotation, and affine transformation, making it useful for tasks like object recognition, image stitching, and 3D reconstruction.

---

## 🔧 Installation

### 1. **Install Python and Pip:**

Ensure Python 3.x is installed on your system. If not, you can download it from [python.org](https://www.python.org/downloads/).

### 2. **Install OpenCV:**

SIFT is part of the OpenCV package, so install OpenCV using pip:

```bash
pip install opencv-python
```

### 3. **Install Dependencies:**

Additionally, you will need **numpy** for array handling. Install it with:

```bash
pip install numpy
```

---

## 🖼️ Images

- **`BOX-SIFT-1.jpg`**: The first image for keypoint detection.
- **`BOX-SIFT-2.jpg`**: The second image used to demonstrate keypoint matching.

Make sure these images are placed in the `Images` folder.

---

## ⚙️ Code Explanation

### 1. **Image Loading and Preprocessing:**

The images are loaded in grayscale using OpenCV’s `imread()` function. The images are processed to detect and match the keypoints between them.

```python
# Import necessary libraries
import cv2
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image paths
img1_path = os.path.join(base_path, 'Images', 'BOX-SIFT-1.jpg')
img2_path = os.path.join(base_path, 'Images', 'BOX-SIFT-2.jpg')

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
```

---

### 2. **SIFT Keypoint Detection:**

- **SIFT Initialization**: The `cv2.SIFT_create()` method initializes the SIFT detector.
- **Keypoint Detection and Descriptor Computation**: The `detectAndCompute()` method detects keypoints and computes their descriptors for both images.

```python
# Create the SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
```

---

### 3. **Descriptor Matching:**

- **Brute-Force Matcher**: The `cv2.BFMatcher()` is used to match the descriptors of the two images.
- **Lowe's Ratio Test**: A ratio of 0.75 is used to filter the best matches based on the distance between the descriptors.

```python
# Create a brute-force matcher and apply Lowe's ratio test to find good matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # Find the best two matches for each descriptor
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Lowe's ratio test
```

---

### 4. **Displaying Matches:**

- **Draw Matches**: The matches between keypoints are visualized using `cv2.drawMatches()`, which connects matching points between the two images.
- **Save the Result**: The resulting image is saved to the **Results** folder.

```python
# Draw the matches between keypoints in both images
result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the result image
cv2.imwrite(output_path, result_img)
```

---

## 🖥️ **How to Run**

1. Place your images in the `Images` folder.
2. Ensure the script is placed in the `SIFT` folder.
3. Run the script using the following command:

   ```bash
   python sift_match.py
   ```

4. The output result will be saved in the **Results** folder as `sift_result.jpg`.

---

## 🛠️ **Troubleshooting**

- **Image Loading Error**: Make sure the image paths are correct and that the images exist in the `Images` folder.
- **Empty Output Image**: If the output is blank, check if the keypoints are being detected by printing the length of the keypoints:

  ```python
  print(f"Keypoints in image 1: {len(kp1)}")
  print(f"Keypoints in image 2: {len(kp2)}")
  ```

- **Missing Dependencies**: Ensure all dependencies are installed by running:

  ```bash
  pip install opencv-python numpy
  ```

---

## 📂 **Folder Structure Recap**

- **Images Folder**: Contains the input images (`BOX-SIFT-1.jpg` and `BOX-SIFT-2.jpg`).
- **Results Folder**: Stores the output result (`sift_result.jpg`).
- **sift_match.py**: The Python script containing the logic for detecting and matching keypoints using SIFT.

---

## 🔄 **Alternative: SURF Implementation**

If you're interested in using the **SURF** (Speeded-Up Robust Features) algorithm instead of SIFT, refer to the **SURF** README file for an alternative implementation.

---

## 📚 **References**

- [SIFT - Scale Invariant Feature Transform](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🎉 **Enjoy working with the SIFT Algorithm!**

Feel free to explore, modify, and experiment with this code. For questions or issues, don’t hesitate to open an issue or contact the author. 😊

---
