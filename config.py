# ============================================================================
# config.py
# ============================================================================
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

IMAGES_DIR = os.path.join(TEMP_DIR, "images")
PREPROCESSED_DIR = os.path.join(TEMP_DIR, "preprocessed")
CELLS_DIR = os.path.join(TEMP_DIR, "cells")

MODEL_PATH = os.path.join(BASE_DIR, "models", "ocr_classifier_final.h5")
# config.py

LLM_MODEL_PATH = r"D:\\llm_check\\granite-3.3-2b-instruct-Q3_K_L.gguf"

# Settings
PDF_DPI = 300
TABLE_ROWS = 8
TABLE_COLS = 5
CELL_MARGIN = 10

# Create directories
for directory in [INPUT_DIR, TEMP_DIR, OUTPUT_DIR, IMAGES_DIR, PREPROCESSED_DIR, CELLS_DIR]:
    os.makedirs(directory, exist_ok=True)
