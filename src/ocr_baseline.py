# src/ocr_baseline.py
"""
OCR baseline: run Tesseract directly on raw document images, independent
of FUNSD's ground-truth annotations. This is the same code we'll reuse
later on the (unlabeled, real-world) FIR dataset — so no FUNSD-specific
shortcuts here, treat the image as if we know nothing about it.
"""

import pytesseract
from pathlib import Path
from PIL import Image

# Uncomment and set this if pytesseract can't find the Tesseract binary automatically (common on Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "data/funsd/training_data/images"


def run_ocr(image: Image.Image):
    """
    Run Tesseract OCR on a PIL image and extract word-level text + bounding boxes.

    Tesseract's image_to_data() gives per-word output in a structured dict,
    including position (left, top, width, height) and a confidence score.
    We convert (left, top, width, height) -> (x0, y0, x1, y1) to match the
    box format used in data_utils.py, so both pipelines are directly comparable.

    Returns:
        words: list[str]
        boxes: list[list[int]]  - [x0, y0, x1, y1]
        confidences: list[float] - per-word OCR confidence (0-100), useful later
                                    for filtering low-confidence detections
    """
    # output_type=DICT gives us a dict of parallel lists (one entry per detected text region)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words, boxes, confidences = [], [], []

    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i].strip()

        # Tesseract returns empty strings for regions it detected but couldn't
        # read confidently (e.g. whitespace gaps) - skip these
        if not text:
            continue

        left = ocr_data["left"][i]
        top = ocr_data["top"][i]
        width = ocr_data["width"][i]
        height = ocr_data["height"][i]

        # convert to (x0, y0, x1, y1) box format
        box = [left, top, left + width, top + height]

        words.append(text)
        boxes.append(box)
        confidences.append(float(ocr_data["conf"][i]))

    return words, boxes, confidences


if __name__ == "__main__":
    stem = "0000971160"
    img_path = next(IMG_DIR.glob(f"{stem}.*"))
    image = Image.open(img_path).convert("RGB")

    words, boxes, confidences = run_ocr(image)

    print(f"Tesseract detected {len(words)} words")
    print("First 5:", list(zip(words[:5], boxes[:5], confidences[:5])))