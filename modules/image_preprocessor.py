# modules/image_preprocessor.py
import cv2
import numpy as np


def preprocess_image_mem(img):
    """Clean and enhance scanned image (IN-MEMORY)."""

    # Remove shadows (UNCHANGED)
    rgb_planes = cv2.split(img)
    result_planes = []
    for plane in rgb_planes:
        dilated = cv2.dilate(plane, np.ones((25, 25), np.uint8))
        bg = cv2.medianBlur(dilated, 31)
        result_planes.append(255 - cv2.absdiff(plane, bg))

    shadow_free = cv2.merge(result_planes)
    gray = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 9
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thick = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=1)
    final = 255 - thick

    return final
