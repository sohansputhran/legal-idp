
import random

import numpy as np
from torch.utils.data import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)

from data_utils import load_funsd_doc, ANN_DIR

LABEL_LIST = ["question", "answer", "header", "other"]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}

# apply_ocr=False because we're supplying our own words/boxes (FUNSD ground truth),
# not asking the processor to run Tesseract internally
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)


def get_train_val_split(val_ratio=0.15, seed=42):
    """
    Split the 149 FUNSD training docs into train/val subsets.
    FUNSD's official 'testing_data' folder is kept untouched -
    reserved for final reporting only, not used here.
    """
    stems = [f.stem for f in ANN_DIR.glob("*.json")]

    rng = random.Random(seed)  # fixed seed -> reproducible split across runs
    rng.shuffle(stems)

    val_size = int(len(stems) * val_ratio)
    val_stems = stems[:val_size]
    train_stems = stems[val_size:]

    return train_stems, val_stems


class FUNSDLayoutDataset(Dataset):
    """
    Wraps FUNSD documents for LayoutLMv3 token classification.
    Each __getitem__ call loads one document, normalizes its boxes,
    and runs it through the processor to produce model-ready tensors.
    """

    def __init__(self, stems):
        self.stems = stems

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        image, words, boxes, labels = load_funsd_doc(stem)
        label_ids = [label2id[l] for l in labels]

        encoding = processor(
            image,
            words,
            boxes=boxes,
            word_labels=label_ids,
            truncation=True,
            padding="max_length",   # pad to model max length here; simplifies
            max_length=512,          # collator usage for this first pass
            return_tensors="pt",
        )

        # processor adds a batch dim (shape [1, ...]) - squeeze it off, since
        # Dataset __getitem__ must return a single unbatched example
        return {k: v.squeeze(0) for k, v in encoding.items()}


def compute_metrics(eval_pred):
    """
    Converts raw model logits + label IDs into precision/recall/F1 via seqeval.
    Strips -100 positions (padding + non-first sub-word tokens) before scoring,
    since these aren't real predictions being evaluated.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # [batch, seq_len] predicted class IDs

    true_predictions = [
        [id2label[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


if __name__ == "__main__":
    train_stems, val_stems = get_train_val_split()
    print(f"Train docs: {len(train_stems)}, Val docs: {len(val_stems)}")

    train_dataset = FUNSDLayoutDataset(train_stems)
    val_dataset = FUNSDLayoutDataset(val_stems)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL_LIST),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./checkpoints/layoutlmv3-funsd",
        per_device_train_batch_size=2,   # small - 6GB VRAM constraint
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size = 2*4 = 8
        num_train_epochs=10,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                        # mixed precision - fits 6GB VRAM
        logging_steps=10,
        report_to="none",                 # disable wandb/etc auto-logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()