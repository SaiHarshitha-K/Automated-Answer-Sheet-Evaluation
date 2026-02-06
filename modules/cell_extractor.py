# ============================================================================
# modules/cell_extractor.py
# ============================================================================
import cv2
import numpy as np


def order_points(pts):
    """Order points clockwise: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(image, contour):
    """Apply perspective transformation."""
    pts = contour.reshape(4, 2)
    rect = order_points(pts)
    tl, tr, br, bl = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_table_contour(processed_img):
    """Detect main table contour."""
    contours, _ = cv2.findContours(
        processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 10000:
            return approx
    return None


def extract_cells(image, rows=8, cols=5, margin=10):
    """
    Extract individual cells from table.
    INPUT  : preprocessed image (numpy array)
    OUTPUT : {(row, col): cell_image (numpy array)}
    """
    cell_images = {}
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    processed = cv2.medianBlur(binary, 3)

    table_contour = detect_table_contour(processed)
    if table_contour is None:
        raise ValueError("Could not detect table contour")

    warped = perspective_transform(image, table_contour)
    h, w = warped.shape[:2]

    cell_height = h // rows
    cell_width = w // cols

    cell_images = {}

    for row in range(rows):
        for col in range(cols):
            y1 = max(0, row * cell_height - margin)
            y2 = min(h, (row + 1) * cell_height + margin)
            x1 = max(0, col * cell_width - margin)
            x2 = min(w, (col + 1) * cell_width + margin)

            cell = warped[y1:y2, x1:x2]
            cell_images[(row + 1, col + 1)] = cell

    return cell_images
