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

def extract_qa_pairs(stem: str):
    """
    Extract question -> answer text pairs from one FUNSD document,
    using the 'linking' field to resolve which answer belongs to which question.
    """
    ann_path = ANN_DIR / f"{stem}.json"
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    # Step 1: build an id -> entry lookup so we can resolve linked ids
    # back to their actual entries (text, label) in O(1)
    entries_by_id = {entry["id"]: entry for entry in data["form"]}

    qa_pairs = {}

    # Step 2: walk every entry that's labeled a question
    for entry in data["form"]:
        if entry["label"] != "question":
            continue

        question_text = entry["text"]

        # Step 3: check its links - each link is a [id_a, id_b] pair.
        # One of the two ids is this question's own id; the other is
        # whatever it's connected to (usually an answer).
        for link in entry["linking"]:
            other_id = link[0] if link[1] == entry["id"] else link[1]

            linked_entry = entries_by_id.get(other_id)
            if linked_entry is None:
                continue  # defensive - shouldn't happen, but linking data can be messy

            # Step 4: only keep it if the linked entry is actually an answer
            # (a question can link to a header or another question in some docs)
            if linked_entry["label"] == "answer":
                qa_pairs[question_text] = linked_entry["text"]

    return qa_pairs

def pairs_to_donut_target(qa_pairs: dict) -> str:
    """
    Serialize question->answer pairs into Donut's tag-based target format.
    e.g. {"Date:": "9/3/92"} -> "<s_question>Date:</s_question><s_answer>9/3/92</s_answer>"
    """
    parts = []
    for question, answer in qa_pairs.items():
        parts.append(f"<s_question>{question}</s_question><s_answer>{answer}</s_answer>")
    return "".join(parts)

if __name__ == "__main__":
    stem = "0000971160"
    # stem = next(ANN_DIR.glob("*.json")).stem
    image, words, boxes, labels = load_funsd_doc(stem)
    print(f"Image size: {image.size}")
    print(f"Num words: {len(words)}")
    print("First 5:", list(zip(words[:5], boxes[:5], labels[:5])))

    pairs = extract_qa_pairs(stem)
    # for q, a in pairs.items():
    #     print(f"{q!r} -> {a!r}")
    target = pairs_to_donut_target(pairs)
    print(target)