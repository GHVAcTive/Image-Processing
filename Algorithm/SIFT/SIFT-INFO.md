# 🖼️ **SIFT (Scale-Invariant Feature Transform) Keypoint Matching Algorithm**

This repository demonstrates the usage of **SIFT (Scale-Invariant Feature Transform)** to detect and match keypoints between two images. The code reads input images, processes them, and draws the matching keypoints between them. The result is saved as an output image for further analysis or visualization.

---

## 📋 **Table of Contents**

- [🔧 Prerequisites](#prerequisites)
- [📥 Installation](#installation)
- [⚙️ Usage Instructions](#usage-instructions)
- [🧑‍💻 Code Explanation](#code-explanation)
- [📖 Theory Behind SIFT](#theory-behind-sift)
- [📊 Output](#output)
- [⚠️ Troubleshooting](#troubleshooting)
- [📜 License](#license)

---

## 📁 **Project Structure**

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

## 📝 **Overview**

This project implements the **SIFT (Scale-Invariant Feature Transform)** algorithm to detect and match keypoints between two images. It uses the **OpenCV** library to perform keypoint detection, descriptor extraction, and matching, providing a simple implementation of SIFT keypoint matching. SIFT is robust to changes in scale, rotation, and affine transformation, making it useful for tasks like object recognition, image stitching, and 3D reconstruction.

---

## 🔧 **Prerequisites**

Before running the code, ensure that you have the following installed:

- **Python** (version 3.6 or higher)
- **OpenCV** library (for computer vision operations)

You can install OpenCV by running:

```bash
pip install opencv-python
```

---

## 📥 **Installation**

### 1. Clone or download the repository

Clone the repository to your local machine or navigate to the project folder:

```bash
git clone <repository_url>
```

### 2. Set up the **Images** folder

Place the images you want to process into the **Images** folder within this repository. For this specific example, you should have the following images in **PNG/JPG** format:

- `BOX-SIFT-1.jpg`
- `BOX-SIFT-2.jpg`

### 3. Run the script

The script will automatically look for these images in the **Images** folder. It will also generate the **Results** folder to store the output image.

---

## ⚙️ **Usage Instructions**

### Running the Script

1. Navigate to the **SIFT** folder containing the Python script:

   ```bash
   cd Algorithm/SIFT
   ```

2. Run the `sift_match.py` script:

   ```bash
   python sift_match.py
   ```

   The script performs the following tasks:

   - **Loads two images** from the **Images** folder (make sure the images are named correctly).
   - **Detects keypoints** in both images using **SIFT**.
   - **Matches keypoints** using **Brute-Force Matching** with Lowe's ratio test.
   - **Draws lines** between the matching keypoints.
   - **Saves the result** in the **Results** folder.

3. **Result**: The output image will be saved in the **Results** folder as `sift_result.jpg`.

---

## 🧑‍💻 **Code Explanation**

Here’s a breakdown of the **`sift_match.py`** script:

```python
# Import necessary libraries
import cv2
import os

# Define the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Paths for the images and output image result
img1_path = os.path.join(base_path, 'Images', 'BOX-SIFT-1.jpg')  # Replace with your image name
img2_path = os.path.join(base_path, 'Images', 'BOX-SIFT-2.jpg')  # Replace with your image name

# Create Results folder if it does not exist
results_folder = os.path.join(base_path, 'Results')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Define the output path for the result image
output_path = os.path.join(results_folder, 'sift_result.jpg')

# Load images in grayscale
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # Load first image
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)  # Load second image
```

### Explanation of the code:

- **Import Libraries**: 
  - `cv2` is OpenCV's library for computer vision tasks.
  - `os` is used to handle file paths and directories.

- **Define Image Paths**: 
  - The code reads the input images from the **Images** folder using relative paths. 
  - Make sure to put the images in the **Images** folder and modify their names in the code if necessary.

- **Create the Results Folder**:
  - The **Results** folder is created automatically if it doesn't exist. This is where the final image (showing keypoint matches) will be saved.

---

### Keypoint Detection and Matching

```python
# Create the SIFT detector object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)  # For the first image
kp2, des2 = sift.detectAndCompute(img2, None)  # For the second image
```

- **SIFT Detector**: `cv2.SIFT_create()` is used to create a SIFT detector.
- **Keypoints and Descriptors**: The `detectAndCompute()` function is used to find keypoints and their descriptors. Keypoints are the points of interest in an image that SIFT uses to describe the image features.

---

### Brute-Force Matching with Lowe's Ratio Test

```python
# Create a brute-force matcher and apply Lowe's ratio test to find good matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # Find the best two matches for each descriptor
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]  # Lowe's ratio test
```

- **BFMatcher**: The brute-force matcher compares the descriptors of the two images and finds the best matches.
- **Lowe's Ratio Test**: A ratio of 0.75 is used to filter good matches by comparing the first and second closest matches.

---

### Draw Matches and Save Result

```python
# Draw matches between keypoints in both images
result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Save the result
cv2.imwrite(output_path, result_img)
print(f"[✔] SIFT result saved at: {output_path}")
```

- **Draw Matches**: This function visualizes the matching keypoints between the two images by drawing lines between them.
- **Save Image**: The final result is saved in the **Results** folder as `sift_result.jpg`.

---

## 📖 **Theory Behind SIFT**

SIFT (Scale-Invariant Feature Transform) is an algorithm that identifies distinctive local features in an image that are invariant to changes in scale, rotation, and affine transformations. This makes it useful in tasks such as object recognition, image stitching, and 3D reconstruction.

### Key Concepts:

- **Keypoints**: Points of interest in an image that are detected based on local patterns of intensity.
- **Descriptors**: A unique description of the local image patches around keypoints.
- **Matching**: Comparing the descriptors of keypoints from two images to find similarities.

---

## 📊 **Output**

The output image will be stored in the **Results** folder:

- **File Name**: `sift_result.jpg`
- **Location**: `Results/sift_result.jpg`

The image will show the two input images with lines drawn between matching keypoints.

---

## ⚠️ **Troubleshooting**

- **Image Loading Error**: Verify that the image paths are correct and the images exist in the **Images** folder. Ensure that the images are in **.jpg** or **.png** format.
  
- **Empty Output Image**: If the output image is blank, check the number of detected keypoints in each image using:

  ```python
  print(f"Keypoints in image 1: {len(kp1)}")
  print(f"Keypoints in image 2: {len(kp2)}")
  ```

- **Missing Dependencies**: If OpenCV is not installed correctly, install it via:

  ```bash
  pip install opencv-python
  ```

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🎉 **Enjoy working with the SIFT Algorithm!**

Feel free to explore, modify, and experiment with this code. For questions or issues, don’t hesitate to open an issue or contact the author. 😊

--- 