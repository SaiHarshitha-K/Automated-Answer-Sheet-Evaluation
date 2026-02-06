# main.py
import os
import glob
import uuid
import cv2

from config import IMAGES_DIR, MODEL_PATH, PDF_DPI
from modules.pdf_converter import convert_pdf_to_images
from modules.image_preprocessor import preprocess_image_mem
from modules.name_prn_extractor import extract_name_prn
from modules.cell_extractor import extract_cells
#from modules.answer_predictor import predict_cell
from modules.answer_predictor import predict_cells_batch
from modules.name_prn_extractor import extract_name_prn_from_image
from utils.prn_utils import normalize_prn


def process_pdf(pdf_path):
    """
    Process a single PDF end-to-end and return one result dict per page.
    """
    results = []

    pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]
    run_dir = os.path.join(IMAGES_DIR, f"run_{pdf_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(run_dir, exist_ok=True)

    # Convert PDF pages → images (disk kept for UI display)
    image_paths = convert_pdf_to_images(pdf_path, run_dir, PDF_DPI)

    for page_num, image_path in enumerate(image_paths, start=1):
        try:
            # -------------------------------
            # Load page image (ONCE)
            # -------------------------------
            page_img = cv2.imread(image_path)
            if page_img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # -------------------------------
            # Preprocess IN MEMORY
            # -------------------------------
            preprocessed_img = preprocess_image_mem(page_img)

            # -------------------------------
            # Extract Name & PRN (from image)
            # -------------------------------
            name, prn = extract_name_prn_from_image(preprocessed_img)
            safe_prn = normalize_prn(prn)

            # Rename page image for UI clarity
            new_image_path = os.path.join(run_dir, f"{safe_prn}.jpg")
            counter = 1
            while os.path.exists(new_image_path):
                new_image_path = os.path.join(run_dir, f"{safe_prn}_dup{counter}.jpg")
                counter += 1

            os.rename(image_path, new_image_path)
            image_path = new_image_path

            # -------------------------------
            # Extract cells IN MEMORY
            # -------------------------------
            cell_images = extract_cells(preprocessed_img)

            # -------------------------------
            # Predict answers (unchanged logic)
            # -------------------------------
            #answers = {}
            #for qno, ((row, col), cell_img) in enumerate(
            #    sorted(cell_images.items()), start=1
            #):
            #    answers[qno] = predict_cell(cell_img)

            # cell_images keys are (row, col) like (1,1), (1,2) ...
            cell_images_by_qno = {}
            qno = 1
            for row in range(1, 9):       # 8 rows
                for col in range(1, 6):   # 5 cols
                    cell_images_by_qno[qno] = cell_images[(row, col)]
                    qno += 1

            answers = predict_cells_batch(cell_images_by_qno)


            # -------------------------------
            # Collect result
            # -------------------------------
            results.append(
                {
                    "page": page_num,
                    "name": name,
                    "prn": prn,
                    "answers": answers,
                    "image_path": image_path,  # ORIGINAL page image for UI
                }
            )

            # Optional logging
            print("\n" + "=" * 60)
            print(f"Page {page_num}")
            print(f"Name: {name}")
            print(f"PRN : {prn}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"ERROR on page {page_num}: {e}")
            continue

    return results


def main():
    """Standalone CLI entry (optional)."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    input_dir = "input_pdfs"
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))

    if not pdf_files:
        print("No PDFs found.")
        return

    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}")
        process_pdf(pdf_path)


if __name__ == "__main__":
    main()
