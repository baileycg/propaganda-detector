# Models

The trained model weights are **not stored in this repository** because they exceed
GitHub's 100 MB file size limit (the DistilBERT model is ~235 MB).

---

## Option A — Automatic download (recommended)

From inside the `propaganda_detector/` directory, run:

```bash
python download_model.py
```

This fetches `distilbert_model.zip` from the latest GitHub Release, extracts it
here, and cleans up the zip. Takes ~2 minutes depending on your connection.

---

## Option B — Manual download

1. Go to the [Releases page](https://github.com/baileycg/propaganda-detector/releases)
2. Under the latest release, download **`distilbert_model.zip`**
3. Extract it so that this folder contains `distilbert_model/`:

```
models/
└── distilbert_model/
    ├── config.json
    ├── model.safetensors      ← the weights (~235 MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── trainer_meta.json
```

---

## Option C — Train from scratch (~15–25 min on CPU)

```bash
python train_transformer.py --data data/raw/mbic.csv --final-eval
```

This fine-tunes `distilbert-base-uncased` on the MBIC dataset and saves the
model here automatically.

---

## After you have the model

```bash
# Classify a single sentence
PYTHONUTF8=1 python main.py --model-type transformer --text "your text here"

# Batch classify a CSV
PYTHONUTF8=1 python main.py --model-type transformer --csv your_file.csv --text-col body --output results.csv
```

## Model performance

| Metric | Value |
|--------|-------|
| Accuracy | 76.0% |
| F1 (Biased) | 0.81 |
| ROC-AUC | 0.82 |
| Non-biased recall | 70% |

Trained on the [MBIC corpus](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE)
— 1,551 labeled sentences from 8 US news outlets across 14 political topics.
