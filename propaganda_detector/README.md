# Propaganda / Media Bias Detector

Detects propaganda and biased language in text, social media posts, or web articles using a fine-tuned **DistilBERT** model trained on the [Media Bias Group's MBIC dataset](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE).

## Model performance (held-out test set)

| Metric | SVM baseline | DistilBERT |
|--------|-------------|------------|
| Accuracy | 67.0% | **76.0%** |
| F1 (Biased) | 0.79 | **0.81** |
| ROC-AUC | 0.71 | **0.82** |
| Non-biased recall | 16% | **70%** |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/baileycg/propaganda-detector.git
cd propaganda-detector/propaganda_detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

### 3. Download NLTK resources

```bash
python -c "
import nltk
for r in ['punkt','punkt_tab','stopwords','vader_lexicon',
          'averaged_perceptron_tagger','averaged_perceptron_tagger_eng']:
    nltk.download(r, quiet=True)
print('Done')
"
```

---

## Training

The trained model weights are not stored in git (too large). You must train locally once.

### Option A — Fast sklearn baseline (~30 seconds)

```bash
python train.py --data data/raw/mbic.csv --compare --final-eval
```

### Option B — DistilBERT (recommended, ~15-25 min on CPU)

```bash
python train_transformer.py --data data/raw/mbic.csv --final-eval
```

Both scripts save their model to `models/` automatically.

---

## Running predictions on new data

### Single text string

```bash
# DistilBERT (accurate)
PYTHONUTF8=1 python main.py --model-type transformer --text "Your text here"

# Fast sklearn fallback
PYTHONUTF8=1 python main.py --text "Your text here"
```

### CSV file (batch)

Your CSV needs a column containing the text to classify.

```bash
PYTHONUTF8=1 python main.py \
  --model-type transformer \
  --csv path/to/your_data.csv \
  --text-col <column_name> \
  --output results.csv
```

Output columns added to your CSV:

| Column | Description |
|--------|-------------|
| `label` | `Propaganda / Biased` or `Non-biased` |
| `confidence` | Model confidence 0–1 |
| `probability_biased` | Raw P(biased) — useful for sorting |
| `triggered_words` | Loaded/assertive words found |

### URL (scrape and classify)

```bash
pip install beautifulsoup4
PYTHONUTF8=1 python main.py --model-type transformer --url https://example.com/article
```

### Interactive mode

```bash
PYTHONUTF8=1 python main.py --model-type transformer --interactive
```

### Adjusting sensitivity

Lower the threshold to flag more content; raise it to be more conservative:

```bash
# More aggressive (catches more bias, more false positives)
PYTHONUTF8=1 python main.py --model-type transformer --threshold 0.4 --text "..."

# More conservative (fewer false positives)
PYTHONUTF8=1 python main.py --model-type transformer --threshold 0.65 --text "..."
```

---

## Project structure

```
propaganda_detector/
├── data/
│   ├── raw/mbic.csv              # MBIC dataset (1,700 labeled sentences)
│   └── processed/                # Train/val/test splits + prediction outputs
├── models/                       # Saved model weights (git-ignored)
├── src/
│   ├── data_loader.py            # MBIC dataset loading & splitting
│   ├── features.py               # NLP feature extraction (TF-IDF + linguistic)
│   ├── models.py                 # sklearn classifiers (SVM, RF, etc.)
│   ├── transformer_model.py      # DistilBERT fine-tuning & inference
│   └── predictor.py              # High-level predict API
├── train.py                      # Train sklearn models
├── train_transformer.py          # Fine-tune DistilBERT
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Dataset

The [MBIC corpus](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE) contains 1,700 sentences from 8 US news outlets (Breitbart, Fox News, Reuters, HuffPost, etc.) across 14 political topics, each labeled **Biased** or **Non-biased** by human annotators.
