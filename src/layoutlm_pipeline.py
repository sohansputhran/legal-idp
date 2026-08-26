from transformers import LayoutLMv3Processor
from data_utils import load_funsd_doc

LABEL_LIST = ["question", "answer", "header", "other"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# apply_ocr=False because we're supplying our own words/boxes (FUNSD ground truth),
# not asking the processor to run Tesseract internally
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)

if __name__ == "__main__":
    stem = "0000971160"
    image, words, boxes, labels = load_funsd_doc(stem)
    label_ids = [label2id[l] for l in labels]

    encoding = processor(
        image,
        words,
        boxes=boxes,
        word_labels=label_ids,
        truncation=True,
        return_tensors="pt",
    )

    print("Keys:", encoding.keys())
    for k, v in encoding.items():
        print(k, v.shape)