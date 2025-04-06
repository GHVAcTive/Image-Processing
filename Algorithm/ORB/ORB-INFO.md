# 🖼️ ORB Algorithm - Keypoint Detection and Matching

This repository demonstrates the usage of **ORB (Oriented FAST and Rotated BRIEF)** to detect and match keypoints between two images. The code reads input images, processes them, and draws the matching keypoints between them. The result is saved as an output image for further analysis or visualization.

---

## 📋 Table of Contents

- [🔧 Prerequisites](#prerequisites)
- [📥 Installation](#installation)
- [⚙️ Usage Instructions](#usage-instructions)
- [🧑‍💻 Code Explanation](#code-explanation)
- [📖 Theory Behind ORB](#theory-behind-orb)
- [📊 Output](#output)
- [⚠️ Troubleshooting](#troubleshooting)
- [📜 License](#license)

---

## 📁 Project Structure

```
Image-Processing/
│
├── Algorithm/
│   └── ORB/
│       ├── orb_match.py
│       ├── README.md
│       └── Images/
│           ├── RR-SURF-1.png   # Front view of the car
│           └── RR-SURF-2.webp  # Side view of the car (modified)
└── Results/
    └── orb_result.png
```

---

## 📝 Overview

This project implements the **ORB (Oriented FAST and Rotated BRIEF)** algorithm to detect and match keypoints between two images. It utilizes the **OpenCV** library to perform keypoint detection, descriptor extraction, and matching. ORB is a free alternative to SURF, providing a fast and efficient solution for feature matching without licensing restrictions.

---

## 🔧 Prerequisites

Before running the code, ensure that you have the following installed:

- **Python** (version 3.6 or higher)
- **OpenCV** library (for computer vision operations)

You can install OpenCV by running:

```bash
pip install opencv-python
```

Additionally, install **numpy** for array handling:

```bash
pip install numpy
```

---

## 📥 Installation

### 1. Clone or Download the Repository

Clone the repository to your local machine or navigate to the project folder:

```bash
git clone <repository_url>
```

### 2. Set up the **Images** Folder

Place the images you want to process into the **Images** folder within this repository. For this example, ensure you have the following images in **PNG/WebP** format:

- `RR-SURF-1.png`
- `RR-SURF-2.webp`

### 3. Run the Script

The script will automatically search for these images in the **Images** folder. It will also generate a **Results** folder to store the output image.

---

## ⚙️ Usage Instructions

### Running the Script

1. Navigate to the **ORB** folder containing the Python script:

   ```bash
   cd Algorithm/ORB
   ```

2. Run the `orb_match.py` script:

   ```bash
   python orb_match.py
   ```

   The script performs the following tasks:

   - **Loads two images** from the **Images** folder (ensure the image filenames are correct).
   - **Detects keypoints** in both images using the ORB detector.
   - **Matches keypoints** using a Brute-Force matcher with the Hamming distance metric and cross-checking.
   - **Sorts matches** based on the descriptor distance.
   - **Draws lines** between matching keypoints.
   - **Saves the result** in the **Results** folder.

3. **Result**: The output image will be saved in the **Results** folder as `orb_result.png`.

---

## 🧑‍💻 Code Explanation

Below is a breakdown of the `orb_match.py` script:

```python
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

# Create ORB detector with a maximum of 400 keypoints
orb = cv2.ORB_create(400)

# Detect and compute descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2_resized, None)

# Match descriptors using Brute-Force and apply cross-check
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance in ascending order
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches between keypoints in both images
result_img = cv2.drawMatches(
    img1, kp1, img2_resized, kp2, matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save the result image
cv2.imwrite(output_path, result_img)
print(f"[✔] ORB result saved at: {output_path}")
```

### Explanation of the Code:

- **Import Libraries**:  
  `cv2` is used for computer vision operations, and `os` manages file paths and directories.

- **Define Image Paths**:  
  The script constructs absolute paths for the images located in the **Images** folder.

- **Results Folder Creation**:  
  If the **Results** folder does not exist, it is created automatically.

- **Image Loading**:  
  Images are loaded in grayscale for efficient processing.

- **Preprocessing**:  
  The second image is resized to match the first image's dimensions.

- **ORB Detector**:  
  An ORB detector is created using `cv2.ORB_create(400)`, which limits detection to 400 keypoints.

- **Keypoint Detection and Descriptor Computation**:  
  The ORB detector detects keypoints and computes descriptors for both images.

- **Descriptor Matching**:  
  A Brute-Force matcher using the Hamming distance metric matches descriptors, with results sorted by distance.

- **Visualization**:  
  Matched keypoints are drawn between the images using `cv2.drawMatches()`.

- **Saving the Result**:  
  The final output image is saved in the **Results** folder as `orb_result.png`.

---

## 📖 Theory Behind ORB

**ORB (Oriented FAST and Rotated BRIEF)** is a fast, efficient, and free algorithm for feature detection and description. It combines the FAST keypoint detector and the BRIEF descriptor with modifications to achieve rotation invariance and improved performance. ORB is widely used as a free alternative to patented algorithms like SIFT and SURF.

### Key Concepts:

- **Keypoints**: Points of interest in the image detected by FAST.
- **Descriptors**: Binary strings generated by BRIEF, which are robust to noise and computationally efficient.
- **Matching**: The process of comparing descriptors to establish correspondences between keypoints in different images.

---

## 📊 Output

The output image will be stored in the **Results** folder:

- **File Name**: `orb_result.png`
- **Location**: `Results/orb_result.png`

The resulting image displays the two input images with lines drawn between matching keypoints.

---

## ⚠️ Troubleshooting

- **Image Loading Error**:  
  Ensure that the image paths are correct and that the images exist in the **Images** folder. The images should be in **PNG**, **JPG**, or **WebP** format.

- **Empty Output Image**:  
  If the output image appears blank, print the number of detected keypoints:
  
  ```python
  print(f"Keypoints in image 1: {len(kp1)}")
  print(f"Keypoints in image 2: {len(kp2)}")
  ```
  
- **Missing Dependencies**:  
  If OpenCV or numpy is not installed, run:
  
  ```bash
  pip install opencv-python numpy
  ```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🎉 Enjoy working with the ORB Algorithm!

Feel free to explore, modify, and experiment with this code. For questions or issues, please open an issue or contact the author. 😊

---