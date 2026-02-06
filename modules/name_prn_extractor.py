# ============================================================================
# modules/name_prn_extractor.py
# ============================================================================

# ============================================================================
# modules/name_pnr_extractor.py
# ============================================================================
import cv2
import numpy as np
from doctr.models import ocr_predictor
from PIL import Image
import streamlit as st

import re

# ---------------------------------------------------------------------------
# GLOBAL DOCTR MODEL (LOADED ONCE)
# ---------------------------------------------------------------------------
#_doctr_model = None


#def get_doctr_model():
#    """Return a singleton DocTR OCR predictor."""
#    global _doctr_model
#    if _doctr_model is None:
#        print("Loading DocTR OCR model for Name/PRN extraction...")
#        _doctr_model = ocr_predictor(pretrained=True)
#    return _doctr_model

@st.cache_resource(show_spinner="Loading DocTR OCR model for Name/PRN...")
def get_doctr_model():
    """Return cached DocTR OCR predictor (loads once per Streamlit app)."""
    print("Loading DocTR OCR model for Name/PRN extraction... (first time only)")
    return ocr_predictor(pretrained=True)



# ---------------------------------------------------------------------------
# BOX DETECTION (ROBUST) 
# ---------------------------------------------------------------------------
def detect_name_prn_box_robust(image, debug=False):
    """
    Detect the Name/PRN box (2nd box from the top) using OpenCV.
    Returns: cropped_box, (x, y, w, h)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    H, W = gray.shape

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    horizontal = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)

    combined = cv2.bitwise_or(horizontal, vertical)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if (
            y < H * 0.5 and      # upper half of page
            w > W * 0.4 and      # quite wide
            h < H * 0.15 and     # not huge vertically
            h > 30 and
            area > W * 50
        ):
            boxes.append({"x": x, "y": y, "w": w, "h": h})

    boxes.sort(key=lambda b: b["y"])

    if debug:
        print(f"[BOX] Found {len(boxes)} candidate boxes")

    if len(boxes) >= 2:
        box = boxes[1]
        if debug:
            print("[BOX] Using 2nd box (Name/PRN line)")
    elif len(boxes) == 1:
        box = boxes[0]
        if debug:
            print("[BOX] Only one box found, using that")
    else:
        # Fallback region if detection fails
        if debug:
            print("[BOX] No box found, using fallback region")
        box = {"x": 80, "y": 200, "w": W - 160, "h": 80}

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)

    cropped = image[y:y + h, x:x + w]
    return cropped, (x, y, w, h)


# ---------------------------------------------------------------------------
# PREPROCESSING FOR OCR
# ---------------------------------------------------------------------------
def preprocess_for_ocr(image_region):
    """
    Enhance a cropped region before DocTR.
    Returns a single-channel (grayscale) image.
    """
    if len(image_region.shape) == 3:
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_region.copy()

    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    kernel = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]]
    )
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    binary = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        10,
    )

    # Ensure black text over white background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


# ---------------------------------------------------------------------------
# OCR ON ORIGINAL + PREPROCESSED BOX
# ---------------------------------------------------------------------------
def extract_text_whole_box_doctr(box_image, debug=False):
    """
    Run DocTR on:
      - original cropped box
      - preprocessed version of the same box

    Returns a dict:
      {
        "doctr_original": [ { "text": ..., "y": ..., "geometry": ... }, ... ],
        "doctr_preprocessed": [ ... ]
      }
    """
    model = get_doctr_model()
    results = {}

    # --- original box ---
    try:
        if len(box_image.shape) == 3:
            pil_img = Image.fromarray(
                cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB)
            )
        else:
            pil_img = Image.fromarray(
                cv2.cvtColor(box_image, cv2.COLOR_GRAY2RGB)
            )

        doc_result = model([np.array(pil_img)])
        lines_with_pos = []

        for page in doc_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(w.value for w in line.words)
                    y_pos = line.geometry[0][1]
                    lines_with_pos.append(
                        {
                            "text": text,
                            "y": y_pos,
                            "geometry": line.geometry,
                        }
                    )

        results["doctr_original"] = lines_with_pos
    except Exception as e:
        if debug:
            print(f"[DOCTR] Error on original box: {e}")
        results["doctr_original"] = []

    # --- preprocessed box ---
    try:
        preprocessed = preprocess_for_ocr(box_image)
        pil_img_p = Image.fromarray(
            cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        )
        doc_result = model([np.array(pil_img_p)])

        lines_with_pos = []
        for page in doc_result.pages:
            for block in page.blocks:
                for line in block.lines:
                    text = " ".join(w.value for w in line.words)
                    y_pos = line.geometry[0][1]
                    lines_with_pos.append(
                        {
                            "text": text,
                            "y": y_pos,
                            "geometry": line.geometry,
                        }
                    )

        results["doctr_preprocessed"] = lines_with_pos
    except Exception as e:
        if debug:
            print(f"[DOCTR] Error on preprocessed box: {e}")
        results["doctr_preprocessed"] = []

    if debug:
        for method, lines in results.items():
            print(f"\n[{method}]")
            for i, line in enumerate(lines):
                print(f"  Line {i+1} (y={line['y']:.2f}): {line['text']}")

    return results


# ---------------------------------------------------------------------------
# PARSE NAME + PRN FROM OCR RESULTS
# ---------------------------------------------------------------------------
def parse_name_prn_from_doctr(ocr_results, debug=False):
    """
    Extract Name and PRN from DocTR outputs coming from both
    original and preprocessed box.
    """
    name_candidates = []
    prn_candidates = []
    seen_prn_texts = set()

    for method in ("doctr_original", "doctr_preprocessed"):
        lines = ocr_results.get(method) or []
        if not lines:
            continue

        if debug:
            print(f"\n[PARSE] Reading lines from {method}")

        # IMPORTANT: keep DocTR order; do NOT sort
        for idx, line in enumerate(lines):
            raw_text = line["text"].strip()
            lower = raw_text.lower()

            # ---- name ----
            if "name" in lower:
                clean_name = re.sub(
                    r"name\s*:?\s*", "", raw_text, flags=re.IGNORECASE
                )
                clean_name = re.sub(
                    r"[^a-zA-Z0-9\s]", "", clean_name
                ).strip()

                if clean_name:
                    name_candidates.append(
                        {
                            "text": clean_name,
                            "method": method,
                            "confidence": len(clean_name),
                        }
                    )
                    if debug:
                        print(f"  -> Name candidate: '{clean_name}'")

            # ---- prn ----
            if "prn" in lower:
                # Same line
                prn_same = re.sub(
                    r"prn\s*:?\s*", "", raw_text, flags=re.IGNORECASE
                )
                prn_same = re.sub(r"[^a-zA-Z0-9]", "", prn_same).strip()

                if re.search(r"\d", prn_same):
                    if prn_same not in seen_prn_texts:
                        seen_prn_texts.add(prn_same)
                        prn_candidates.append(
                            {
                                "text": prn_same,
                                "method": method,
                                "confidence": len(prn_same),
                            }
                        )
                        if debug:
                            print(
                                f"  -> PRN candidate (same line): "
                                f"'{prn_same}'"
                            )
                else:
                    # Next line
                    if idx + 1 < len(lines):
                        next_raw = lines[idx + 1]["text"].strip()
                        next_clean = re.sub(
                            r"[^a-zA-Z0-9]", "", next_raw
                        ).strip()

                        if re.search(r"\d", next_clean):
                            if next_clean not in seen_prn_texts:
                                seen_prn_texts.add(next_clean)
                                prn_candidates.append(
                                    {
                                        "text": next_clean,
                                        "method": method,
                                        "confidence": len(next_clean),
                                    }
                                )
                                if debug:
                                    print(
                                        "  -> PRN candidate "
                                        f"(next line): '{next_clean}'"
                                    )

            # fallback numeric-only line
            elif re.search(r"\d{6,}", raw_text):
                alt = re.sub(r"[^a-zA-Z0-9]", "", raw_text).strip()
                if re.search(r"\d", alt) and alt not in seen_prn_texts:
                    seen_prn_texts.add(alt)
                    prn_candidates.append(
                        {
                            "text": alt,
                            "method": method,
                            "confidence": len(alt),
                        }
                    )
                    if debug:
                        print(
                            "  -> PRN candidate "
                            f"(number-only line): '{alt}'"
                        )

    # choose best name
    final_name = ""
    if name_candidates:
        name_candidates.sort(
            key=lambda c: c["confidence"], reverse=True
        )
        final_name = name_candidates[0]["text"]
        if debug:
            print(
                f"\n[FINAL] Name: '{final_name}' "
                f"(from {name_candidates[0]['method']})"
            )

    # choose best prn
    final_prn = ""
    if prn_candidates:
        def prn_score(c):
            digit_count = len(re.findall(r"\d", c["text"]))
            return digit_count, c["confidence"]

        prn_candidates.sort(key=prn_score, reverse=True)
        final_prn = prn_candidates[0]["text"]
        if debug:
            print(
                f"[FINAL] PRN (raw): '{final_prn}' "
                f"(from {prn_candidates[0]['method']})"
            )

    return final_name, final_prn


# ---------------------------------------------------------------------------
# NORMALISE PRN (FIX O/0, S/5, ETC.)
# ---------------------------------------------------------------------------
def normalise_prn_text(prn_raw):
    """
    Convert typical OCR letter mistakes to digits and keep only numbers.
    Examples:
      O -> 0,  S -> 5,  I/l -> 1,  B -> 8, etc.
    """
    if not prn_raw:
        return ""

    mapping = {
        "O": "0", "o": "0",
        "I": "1", "l": "1",
        "Z": "2", "z": "2",
        "S": "5", "s": "5",
        "B": "8", "b": "8",
    }

    corrected_chars = []
    for ch in prn_raw:
        if ch in mapping:
            corrected_chars.append(mapping[ch])
        else:
            corrected_chars.append(ch)

    corrected = "".join(corrected_chars)
    digits_only = "".join(c for c in corrected if c.isdigit())

    return digits_only or corrected


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def extract_name_prn(image_path, debug=False):
    """
    Main function used by main.py.
    Takes a path to the page image, detects the Name/PRN box,
    runs DocTR on both original and preprocessed box, and returns:

        name, prn

    where prn is normalised to numeric form.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    box_crop, _ = detect_name_prn_box_robust(img, debug=debug)
    ocr_results = extract_text_whole_box_doctr(box_crop, debug=debug)
    name_raw, prn_raw = parse_name_prn_from_doctr(ocr_results, debug=debug)

    prn_final = normalise_prn_text(prn_raw)
    return name_raw, prn_final


def extract_name_prn_from_image(image, debug=False):
    """
    In-memory version of extract_name_prn.
    BEHAVIOR IDENTICAL — only skips cv2.imread.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image passed to extract_name_prn_from_image")

    # USE IMAGE DIRECTLY (no disk)
    box_crop, _ = detect_name_prn_box_robust(image, debug=debug)
    ocr_results = extract_text_whole_box_doctr(box_crop, debug=debug)
    name_raw, prn_raw = parse_name_prn_from_doctr(ocr_results, debug=debug)

    prn_final = normalise_prn_text(prn_raw)
    return name_raw, prn_final
