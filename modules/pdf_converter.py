# ============================================================================
# modules/pdf_converter.py
# ============================================================================
import fitz
import os


def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    """Convert PDF pages to images."""
    doc = fitz.open(pdf_path)
    image_paths = []
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.jpg")
        pix.save(output_path)
        image_paths.append(output_path)
    
    doc.close()
    return image_paths