# 🎯 **Harris Corner Detection Algorithm** 🖼️🔍

This repository demonstrates how to apply the **Harris Corner Detection** technique on an image to detect corners and highlight them in **Blue**. This algorithm is essential for detecting key points and edges in an image, which is widely used in computer vision tasks like feature matching, object recognition, and tracking.

---

## 📋 **Table of Contents**

- [🔧 Prerequisites](#prerequisites)
- [📥 Installation](#installation)
- [⚙️ Usage Instructions](#usage-instructions)
- [🧑‍💻 Code Explanation](#code-explanation)
- [📊 Output](#output)
- [⚠️ Troubleshooting](#troubleshooting)
- [📜 License](#license)

---

## 📁 **Project Structure**

```
Image-Processing/
│
├── Algorithm/
│   └── Harris_Corner/
│       ├── harris_corner.py
│       ├── README.md
├── Images/
│   └── Geometric_Art-Harris_Corner.jpg
└── Results/
    └── harris_corners_result.png
```

---

## 📝 **Overview**

This project applies the **Harris Corner Detection** algorithm to detect and highlight the corners in an image. The corners are marked in **Blue** and are dilated for increased visibility, making them appear larger and more prominent.

---

## 🔧 **Prerequisites**

Ensure you have the following installed:

- **Python 3.6+**
- **OpenCV**  
  Install it via:

  ```bash
  pip install opencv-python opencv-contrib-python
  ```

- **NumPy**  
  Install with:

  ```bash
  pip install numpy
  ```

---

## 📥 **Installation**

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd Image-Processing
   ```

2. **Set up the Images Folder:**

   Place the image (`Geometric_Art-Harris_Corner.jpg`) in the `Images` folder.

3. **Run the Script** after setting up the folder:

   ```bash
   python Algorithm/Harris/harris_corner.py
   ```

---

## ⚙️ **Usage Instructions**

1. **Navigate to the Harris Corner Folder:**

   ```bash
   cd Algorithm/Harris
   ```

2. **Run the Script:**

   ```bash
   python harris_corner.py
   ```

   - The script will load the image, perform **Harris Corner Detection**, and highlight the detected corners in **Blue**.
   - The corners will be dilated to make them **larger and more visible**.

---

## 🧑‍💻 **Code Explanation**

The **`harris_corner.py`** script performs the following steps:

### 1. **Image Loading and Setup**

```python
import cv2
import numpy as np
import os

# Get the absolute path to the project root
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

# Image path using the base path
img_path = os.path.join(base_path, 'Images', 'Geometric_Art-Harris_Corner.jpg')

# Load image in color
img = cv2.imread(img_path)
```

- **Image Loading**:  
  The image is loaded in color using `cv2.imread`.

### 2. **Convert Image to Grayscale**

```python
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert to float32 for Harris corner detection
gray_float32 = np.float32(gray)
```

- **Grayscale Conversion**:  
  The image is converted to grayscale because corner detection works on intensity values, not color channels.

### 3. **Harris Corner Detection**

```python
# Harris corner detection
dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)
```

- **Harris Corner Detection**:  
  The `cv2.cornerHarris` function is used to detect the corners in the grayscale image. The parameters control the neighborhood size, Sobel operator aperture, and Harris parameter.

### 4. **Dilate the Corner Image**

```python
# Dilate the corner image with a larger kernel to make the corners bigger
dst = cv2.dilate(dst, None, iterations=5)
```

- **Dilation**:  
  Dilation is applied to increase the visibility of the corners, making them appear larger.

### 5. **Increase the Corner Size Further and Apply Threshold**

```python
# Apply a threshold to make more corners visible and highlight them
threshold = 0.01 * dst.max()  # Adjust the threshold to select more corners
img[dst > threshold] = [0, 255, 0]  # Color the corners in green (BGR: [0, 255, 0])
```

- **Thresholding**:  
  A threshold is applied to highlight only the most prominent corners. The corners are then colored **green** for better visibility.

### 6. **Display and Save the Result**

```python
# Display the image with corners marked
cv2.imshow('Harris Corner Detection', img)

# Save the result
output_path = os.path.join(base_path, 'Results', 'harris_corners_result.png')
cv2.imwrite(output_path, img)
```

- **Display and Save**:  
  The processed image with the marked corners is displayed, and the result is saved to the `Results` folder.

---

## 📊 **Output**

The output is an image with **corners highlighted in green**. The corners are made **larger and more visible** through dilation.

- **File Name**: `harris_corners_result.png`
- **Location**: `Results/harris_corners_result.png`

The corners will be displayed and saved in the green color you specified.

---

## ⚠️ **Troubleshooting**

- **Image Path Issues**:  
  Ensure the path to the image is correct.
  
- **Corner Visibility**:  
  If the corners are not prominent enough, you can adjust the dilation `iterations` or the `threshold` value.

---

## 📜 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

### 🎉 **Enjoy detecting corners!**

Feel free to experiment and modify the code. If you encounter any issues, don't hesitate to open an issue or contact the project maintainer.

---