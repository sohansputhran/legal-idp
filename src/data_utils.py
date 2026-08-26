# src/data_utils.py

import json
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
ANN_DIR = BASE_DIR / "data/funsd/training_data/annotations"
IMG_DIR = BASE_DIR / "data/funsd/training_data/images"


def load_funsd_doc(stem: str):
    """
    Load one FUNSD document's image + word-level annotations.

    Args:
        stem: filename without extension (e.g. "0000971160")

    Returns:
        image: PIL.Image
        words: list[str]        - individual word text
        boxes: list[list[int]]  - [x0, y0, x1, y1] per word, original pixel scale
        labels: list[str]       - entity label ("question"/"answer"/"header"/"other")
                                   repeated per word (word inherits its parent entry's label)
    """
    img_path = next(IMG_DIR.glob(f"{stem}.*"))
    image = Image.open(img_path).convert("RGB")

    ann_path = ANN_DIR / f"{stem}.json"
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    words, boxes, labels = [], [], []
    for entry in data["form"]:
        label = entry["label"]
        for word in entry["words"]:
            words.append(word["text"])
            boxes.append(word["box"])
            labels.append(label)

    boxes = [normalize_box(box, image.width, image.height) for box in boxes]
    
    return image, words, boxes, labels

def normalize_box(box, width, height):
    """Scale a [x0,y0,x1,y1] pixel box to LayoutLMv3's expected 0-1000 range."""
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

if __name__ == "__main__":
    stem = "0000971160"
    # stem = next(ANN_DIR.glob("*.json")).stem
    image, words, boxes, labels = load_funsd_doc(stem)
    print(f"Image size: {image.size}")
    print(f"Num words: {len(words)}")
    print("First 5:", list(zip(words[:5], boxes[:5], labels[:5])))