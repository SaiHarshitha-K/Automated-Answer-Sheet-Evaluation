# ============================================================================
# modules/answer_predictor.py
# ============================================================================
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

MODEL_PATH = "models/ocr_classifier_final.h5"

@st.cache_resource(show_spinner="Loading answer OCR model...")
def load_answer_model():
    return load_model(MODEL_PATH)
# Load model ONCE
_model = load_model(MODEL_PATH)

CLASS_NAMES = ["A", "B", "C", "D", "scribble"]
SCRIBBLE_LABEL = "scribble"
HIGH_CONF = 0.70

MIN_AREA_FRAC = 0.0015
MAX_AREA_FRAC = 0.35
INNER_MARGIN_FRAC = 0.07
NUM_REGION_X_FRAC = 0.25
NUM_REGION_Y_FRAC = 0.60
MAX_NUM_H_FRAC = 0.20
MAX_NUM_W_FRAC = 0.20
MAX_NUM_AREA_FRAC = 0.03


# ------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------
def segment_cell(cell_img):
    gray = (
        cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        if cell_img.ndim == 3 and cell_img.shape[2] == 3
        else cell_img.copy()
    )

    H, W = gray.shape
    cell_area = H * W

    mh = int(H * INNER_MARGIN_FRAC)
    mw = int(W * INNER_MARGIN_FRAC)
    inner = gray[mh:H - mh, mw:W - mw]

    blur = cv2.GaussianBlur(inner, (5, 5), 0)
    _, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        if area < MIN_AREA_FRAC * cell_area or area > MAX_AREA_FRAC * cell_area:
            continue

        x_full = x + mw
        y_full = y + mh

        in_top_left = (
            x_full < NUM_REGION_X_FRAC * W
            and y_full < NUM_REGION_Y_FRAC * H
        )

        small_like_num = (
            h < MAX_NUM_H_FRAC * H
            and w < MAX_NUM_W_FRAC * W
            and area < MAX_NUM_AREA_FRAC * cell_area
        )

        if in_top_left and small_like_num:
            continue

        candidates.append(
            {
                "x": x_full,
                "y": y_full,
                "w": w,
                "h": h,
                "crop": th[y:y + h, x:x + w],
                "contour": c,
            }
        )

    candidates.sort(key=lambda d: d["x"])
    return candidates


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess_for_model(crop, model):
    _, H, W, C = model.input_shape
    resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
    img = resized.astype("float32") / 255.0
    img = img[..., np.newaxis] if C == 1 else np.stack([img] * 3, axis=-1)
    return np.expand_dims(img, 0)


# ------------------------------------------------------------------
# Scribble heuristics
# ------------------------------------------------------------------
def is_scribble_geom(blob):
    ink = blob["ink_ratio"]
    aspect = blob["aspect"]
    peri = cv2.arcLength(blob["contour"], True)
    diag = np.sqrt(blob["w"] ** 2 + blob["h"] ** 2)

    return (
        ink > 0.65
        or aspect > 4.5
        or aspect < 0.20
        or (diag > 0 and peri / diag > 8.0)
    )


"""def has_diagonal_line(blob):
    edges = cv2.Canny(blob["crop"], 50, 150)
    min_len = int(0.7 * max(blob["h"], blob["w"]))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 20,
        minLineLength=min_len,
        maxLineGap=5
    )

    if lines is None:
        return False

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (25 <= angle <= 65) or (115 <= angle <= 155):
            return True

    return False
"""

def has_diagonal_line(blob):
    crop = blob["crop"]

    # crop is a binary image (0/255). Get size.
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return False

    # ✅ NEW: downscale before Canny + Hough for speed
    max_dim = max(h, w)
    if max_dim > 64:
        scale = 64.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]

    edges = cv2.Canny(crop, 50, 150)

    # recompute min_len after scaling
    min_len = int(0.7 * max(h, w))

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 20,
        minLineLength=min_len,
        maxLineGap=5
    )

    if lines is None:
        return False

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (25 <= angle <= 65) or (115 <= angle <= 155):
            return True

    return False


# ------------------------------------------------------------------
# Prediction (BATCHED, SAFE)
# ------------------------------------------------------------------
def predict_cell(cell_img):
    candidates = segment_cell(cell_img)
    if not candidates:
        return ""

    scrib_idx = CLASS_NAMES.index(SCRIBBLE_LABEL)  # KEEP
    predictions = []

    cnn_inputs = []
    cand_meta = []

    for cand in candidates:
        inp = preprocess_for_model(cand["crop"], _model)
        cnn_inputs.append(inp[0])
        cand_meta.append(cand)

    probs_all = _model.predict(np.array(cnn_inputs), verbose=0)

    for probs, cand in zip(probs_all, cand_meta):
        best_idx = int(np.argmax(probs))
        label = CLASS_NAMES[best_idx]

        blob = {
            "x": cand["x"],
            "center_x": cand["x"] + cand["w"] / 2.0,
            "label": label,
            "prob": float(probs[best_idx]),
            "ink_ratio": float(np.mean(cand["crop"] > 0)),
            "aspect": cand["h"] / max(1, cand["w"]),
            "contour": cand["contour"],
            "w": cand["w"],
            "h": cand["h"],
            "crop": cand["crop"],
        }

        #blob["is_scribble"] = (
        #    label == SCRIBBLE_LABEL
        #    or is_scribble_geom(blob)
        #    or has_diagonal_line(blob)
        #)

        geom_flag = is_scribble_geom(blob)

        if (label == SCRIBBLE_LABEL) or geom_flag or (blob["prob"] < 0.55):
            blob["is_scribble"] = (
                (label == SCRIBBLE_LABEL)
                or geom_flag
                or has_diagonal_line(blob)
            )
        else:
            blob["is_scribble"] = False


        predictions.append(blob)

    non_scrib = [b for b in predictions if not b["is_scribble"]]
    non_scrib.sort(key=lambda b: b["center_x"], reverse=True)

    for b in non_scrib:
        if b["prob"] >= HIGH_CONF:
            return b["label"]

    return non_scrib[0]["label"] if non_scrib else ""



#batch 
def predict_cells_batch(cell_images_by_qno: dict):
    """
    Predict answers for ALL cells in ONE model.predict call.
    Input:  {qno: cell_img}
    Output: {qno: "A"/"B"/"C"/"D"/""}
    """

    batch_imgs = []
    batch_refs = []  # (qno, cand)

    # Collect all candidate blobs from all cells
    for qno, cell_img in cell_images_by_qno.items():
        candidates = segment_cell(cell_img)
        for cand in candidates:
            inp = preprocess_for_model(cand["crop"], _model)
            batch_imgs.append(inp[0])
            batch_refs.append((qno, cand))

    if not batch_imgs:
        return {qno: "" for qno in cell_images_by_qno.keys()}

    # ONE model call
    probs_all = _model.predict(np.array(batch_imgs), verbose=0)

    # Build blobs per cell
    per_cell_preds = {qno: [] for qno in cell_images_by_qno.keys()}

    for probs, (qno, cand) in zip(probs_all, batch_refs):
        best_idx = int(np.argmax(probs))
        label = CLASS_NAMES[best_idx]
        prob = float(probs[best_idx])

        blob = {
            "x": cand["x"],
            "center_x": cand["x"] + cand["w"] / 2.0,
            "label": label,
            "prob": prob,
            "ink_ratio": float(np.mean(cand["crop"] > 0)),
            "aspect": cand["h"] / max(1, cand["w"]),
            "contour": cand["contour"],
            "w": cand["w"],
            "h": cand["h"],
            "crop": cand["crop"],
        }

        # KEEP your scribble logic, but make diagonal check conditional
        geom_flag = is_scribble_geom(blob)
        if (label == SCRIBBLE_LABEL) or geom_flag or (prob < 0.55):
            blob["is_scribble"] = (
                (label == SCRIBBLE_LABEL)
                or geom_flag
                or has_diagonal_line(blob)
            )
        else:
            blob["is_scribble"] = False

        per_cell_preds[qno].append(blob)

    # Apply same selection logic per cell
    answers = {}
    for qno, blobs in per_cell_preds.items():
        non_scrib = [b for b in blobs if not b["is_scribble"]]
        non_scrib.sort(key=lambda b: b["center_x"], reverse=True)

        # high confidence first
        ans = ""
        for b in non_scrib:
            if b["prob"] >= HIGH_CONF:
                ans = b["label"]
                break

        if not ans and non_scrib:
            ans = non_scrib[0]["label"]

        answers[qno] = ans

    return answers
