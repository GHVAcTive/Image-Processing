# 🖼️ **SURF Algorithm – Keypoint Detection and Matching**

This repository demonstrates the usage of **SURF (Speeded-Up Robust Features)** to detect and match keypoints between two images. The code reads input images, processes them using the SURF detector, and draws lines connecting matching keypoints. The final output is saved as an image for further analysis or visualization.

---

## 📋 **Table of Contents**

- [🔧 Prerequisites](#-prerequisites)
- [📥 Installation](#-installation)
- [⚙️ Usage Instructions](#-usage-instructions)
- [🧑‍💻 Code Explanation](#-code-explanation)
- [📖 Theory Behind SURF](#-theory-behind-surf)
- [📊 Output](#-output)
- [⚠️ Troubleshooting](#-troubleshooting)
- [📜 License](#-license)

---

## 📁 **Project Structure**

```
Image-Processing/
│
├── Algorithm/
│   └── SURF/
│       ├── surf_match.py
│       ├── README.md
│       └── Images/
│           ├── RR-SURF-1.png   # Front view of the car
│           └── RR-SURF-2.webp  # Side view of the car (modified)
└── Results/
    └── surf_result.png
```

---

## 📝 **Overview**

This project implements the **SURF (Speeded-Up Robust Features)** algorithm to detect and match keypoints between two images. Utilizing the **OpenCV** library, the code performs keypoint detection, descriptor extraction, and matching between input images. SURF is robust to scale, rotation, and affine transformations—making it ideal for applications such as object recognition and image stitching.

**Note:** SURF is a patented algorithm and is excluded from standard OpenCV builds unless you use the contrib modules. Ensure that you have installed the `opencv-contrib-python` package.

---

## 🔧 **Prerequisites**

Before running the code, ensure that you have the following installed:

- **Python** (version 3.6 or higher)
- **OpenCV** with contrib modules

Install the required packages using:

```bash
pip install opencv-contrib-python numpy
```

---

## 📥 **Installation**

### 1. Clone or Download the Repository

Clone the repository to your local machine or navigate to the project folder:

```bash
git clone <repository_url>
```

### 2. Set Up the **Images** Folder

Place the images you want to process into the **Images** folder within this repository. For this example, ensure you have:

- `RR-SURF-1.png` – Front view of the car
- `RR-SURF-2.webp` – Side view of the car (modified)

### 3. Run the Script

The script will automatically look for these images in the **Images** folder and create a **Results** folder (if not already present) to store the output image.

---

## ⚙️ **Usage Instructions**

### Running the Script

1. Navigate to the **SURF** folder containing the Python script:

   ```bash
   cd Algorithm/SURF
   ```

2. Run the `surf_match.py` script:

   ```bash
   python surf_match.py
   ```

   The script performs the following tasks:
   - **Loads two images** from the **Images** folder.
   - **Detects keypoints** in both images using SURF.
   - **Matches keypoints** using Brute-Force matching with Lowe's ratio test.
   - **Draws lines** between matching keypoints.
   - **Saves the result** in the **Results** folder.

3. **Result**: The output image will be saved in the **Results** folder as `surf_result.png`.

---

## 🧑‍💻 **Code Explanation**

Below is a breakdown of the `surf_match.py` script:

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
kp2, des2 = surf.detectAndCompute(img2_resized, None)  # Use img2_resized or img2_rotated as needed

# Match descriptors using Brute-Force and apply Lowe's ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Draw good matches with custom line color and thickness
result_img = cv2.drawMatches(
    img1, kp1, img2_resized, kp2, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save result
cv2.imwrite(output_path, result_img)
print(f"[✔] SURF result saved at: {output_path}")
```

### Explanation:

- **Importing Libraries**:  
  The script imports `cv2` (OpenCV) and `os` for handling file paths.

- **Defining Image Paths**:  
  It constructs absolute paths for the images located in the **Images** folder.

- **Creating Results Folder**:  
  If the **Results** folder does not exist, it is created automatically.

- **Image Loading**:  
  Images are loaded in grayscale for faster processing.

- **Preprocessing**:  
  The second image is resized to match the dimensions of the first image to simulate changes in scale and orientation.

- **SURF Detector**:  
  The SURF detector is created with a Hessian threshold of 400. This value can be adjusted to control the sensitivity of keypoint detection.

- **Keypoint Detection & Descriptor Extraction**:  
  Keypoints and their descriptors are computed for both images.

- **Descriptor Matching**:  
  A Brute-Force matcher compares the descriptors. Lowe's ratio test is applied to retain only good matches.

- **Visualization**:  
  Matched keypoints are drawn between the two images, and the result is saved.

---

## 📖 **Theory Behind SURF**

**SURF (Speeded-Up Robust Features)** is an algorithm used to detect and describe local features in images. It is designed to be scale- and rotation-invariant and is particularly robust to changes in lighting and viewpoint. SURF is useful for tasks such as object recognition, image stitching, and 3D reconstruction.

### Key Concepts:

- **Keypoints**:  
  Distinctive points in an image, such as corners or blobs, that are detected using the SURF algorithm.

- **Descriptors**:  
  Vectors that describe the local region around each keypoint, allowing for robust matching between images.

- **Matching**:  
  Comparing descriptors from two images to identify corresponding keypoints.

---

## 📊 **Output**

The output image will be stored in the **Results** folder:

- **File Name**: `surf_result.png`
- **Location**: `Results/surf_result.png`

This image displays the two input images with lines drawn between matching keypoints.

---

## ⚠️ **Troubleshooting**

- **Image Loading Error**:  
  Ensure that the image paths are correct and that the images exist in the **Images** folder. Images should be in **PNG**, **JPG**, or **WebP** format.

- **Empty Output Image**:  
  If the output image is blank, verify the number of keypoints detected in each image by printing:

  ```python
  print(f"Keypoints in image 1: {len(kp1)}")
  print(f"Keypoints in image 2: {len(kp2)}")
  ```

- **Missing Dependencies**:  
  If OpenCV or numpy is missing, install them via:

  ```bash
  pip install opencv-contrib-python numpy
  ```

- **SURF Not Available Error**:  
  If you encounter an error such as `cv2.xfeatures2d.SURF_create not found`, ensure that you have installed `opencv-contrib-python` and that your OpenCV build supports non-free algorithms.

---

## 📜 **License**

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

### 🎉 **Enjoy working with the SURF Algorithm!**

Feel free to explore, modify, and experiment with this code. For any questions or issues, please open an issue or contact the author. 😊

---