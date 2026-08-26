# legal-idp

Layout-aware OCR and structured field extraction from scanned/handwritten legal filings (Intelligent Document Processing, IDP). Built as a portfolio project targeting ML Engineer / ML Researcher roles in justice-tech.

## Goal

Extract structured fields (entity type, key-value pairs) from scanned legal documents using layout-aware models (LayoutLMv3, Donut) rather than plain OCR-to-text pipelines — since legal filings depend heavily on spatial structure (forms, stamps, letterheads, handwritten + printed mixed content) that flat text extraction discards.

## Datasets

- **FUNSD** (Form Understanding in Noisy Scanned Documents) — 149 training / 50 test scanned forms, word-level annotations with 4 entity labels (`question`, `answer`, `header`, `other`) and entity-linking pairs. Used for initial pipeline development.
- **FIR dataset** (TransDocAnalyser, ICDAR 2023) — Indian First Information Reports, hybrid printed+handwritten, planned as the real-domain evaluation set.

## Repo structure

```
legal-idp/
  src/
    data_utils.py        # FUNSD loader: image + word-level annotations
    ocr_baseline.py       # Tesseract OCR baseline (engine-only, no layout awareness)
    layoutlm_pipeline.py  # LayoutLMv3 fine-tuning (in progress)
  data/
    funsd/
      training_data/
        annotations/
        images/
```

## Progress log

### EDA (FUNSD)
- Schema: each doc is a list of `form` entries with `box`, `text`, `label`, `words` (sub-word boxes), `linking`, `id`
- Label distribution across 149 training files (7,778 word-level entries): `question` 43.5%, `answer` 38.5%, `other` 12.1%, `header` 5.9% — no severe class imbalance requiring weighting
- Image dimensions consistent: height fixed at 1000px, width 754–863px, no corrupt/outlier files
- Verified annotation boxes fall within image bounds

### `data_utils.py`
Shared loader returning word-level `(image, words, boxes, labels)`, with each word inheriting its parent entry's label (standard FUNSD→LayoutLM convention, since LayoutLMv3 needs one label per token, not per entity span).

### `ocr_baseline.py`
Tesseract-based OCR baseline, run directly on raw images with no dependency on FUNSD ground truth (same code intended for reuse on the unlabeled FIR dataset later).

**Finding:** Tesseract detected 127 words vs. 148 in ground truth on the sample document. Root cause investigated — not a bug, but a real OCR limitation: Tesseract's page segmentation and text recognition are trained on standard printed fonts, and it failed to detect/read a stylized logo ("R&D" letterhead) that FUNSD's human annotators transcribed manually. Documented as a motivating example for why layout-aware / OCR-free models (LayoutLMv3, Donut) are expected to outperform naive OCR-then-classify pipelines on real documents — particularly relevant for Indian FIRs, which contain police station letterheads, stamps, and seals.

**Note:** word-count parity between OCR output and ground truth is not sufficient for validation, since detected words aren't index-aligned across the two sources (different words at the same list position). Formal evaluation will require IoU-based box matching or text-content matching with position tolerance.

### `layoutlm_pipeline.py`
In progress — LayoutLMv3 fine-tuning for token classification (entity labeling) on FUNSD.

## Next steps

- [ ] Build LayoutLMv3 preprocessing (box normalization to 0–1000 scale, processor setup)
- [ ] Fine-tune LayoutLMv3 on FUNSD, evaluate token-level F1
- [ ] Build Donut baseline (OCR-free image→JSON) for comparison
- [ ] Formal OCR-baseline evaluation (IoU-matched word accuracy)
- [ ] Repeat pipeline on FIR dataset (real-world Indian legal documents)
- [ ] Write up comparative results (OCR-only vs LayoutLMv3 vs Donut)

## Stack

- LayoutLMv3, Donut (Hugging Face `transformers`)
- Tesseract (`pytesseract`) for OCR baseline
- No proprietary LLM APIs used