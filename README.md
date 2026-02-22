# propaganda-detector
# Propaganda / Media Bias Detector

Detects propaganda and biased language in text, social media posts, or web articles using a fine-tuned **DistilBERT** model trained on the [MBIC dataset](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE) and augmented with the [PTC v2 corpus](https://zenodo.org/records/3952415).

## Model performance (held-out MBIC test set)

| Metric | SVM baseline | DistilBERT (MBIC only) | DistilBERT + PTC |
|--------|-------------|------------------------|------------------|
| Accuracy | 67.0% | 76.0% | **76.8%** |
| F1 (Biased) | 0.79 | 0.81 | **0.814** |
| ROC-AUC | 0.71 | 0.82 | **0.857** |
| Non-biased recall | 16% | 70% | **76%** |

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
pip install transformers nrclex
pip install spacy && python -m spacy download en_core_web_sm
```

> **GPU (recommended):** Replace the torch install with:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

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

## Getting the model

The weights are found in the releases under v2.0

If you want to recreate the model, you can choose one of the following options

### Option A — Download pre-trained model (recommended, ~2 min)

```bash
python download_model.py
```

Downloads the fine-tuned DistilBERT v2 model (~236 MB) from GitHub Releases into `models/distilbert_model`.

### Option B — Train from scratch

**Fast sklearn baseline (~30 seconds)**
```bash
python train.py --data data/raw/mbic.csv --compare --final-eval
```

**DistilBERT — MBIC only (~ 1-3 hr on CPU, ~5 min on GPU)**
```bash
python train_transformer.py --data data/raw/mbic.csv --final-eval
```

**DistilBERT + PTC v2 augmentation (recommended, best results)**
```bash
python train_transformer.py \
  --data data/raw/mbic.csv \
  --ptc-dir "path/to/datasets-v2/datasets" \
  --final-eval
```

The `--ptc-dir` flag merges ~14,000 PTC v2 propaganda sentences into training, improving AUC from 0.82 to 0.857.

---

## API server

Start the FastAPI backend to serve predictions over HTTP:

```bash
cd propaganda_detector
python api.py
# or: uvicorn api:app --reload --port 8000
```

The server runs at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness check — returns loaded models |
| `GET` | `/models` | List available model backends |
| `POST` | `/predict` | Classify a single text |
| `POST` | `/predict/batch` | Classify up to 100 texts at once |
| `POST` | `/predict/url` | Fetch & classify text from a URL |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The corrupt regime destroyed everything!", "model_type": "transformer", "threshold": 0.5}'
```

Each request accepts an optional `model_type` (`"sklearn"` or `"transformer"`) and `threshold` (0–1).

---

## CLI predictions

### Single text string

```bash
# DistilBERT (most accurate)
python main.py --model-type transformer --model distilbert_model --text "Your text here"

# Fast sklearn fallback
python main.py --text "Your text here"
```

### Interactive mode

```bash
python main.py --model-type transformer --model distilbert_model --interactive
```

Type or paste text at the `Text>` prompt. Enter `quit` to exit.

### CSV file (batch)

```bash
python main.py \
  --model-type transformer \
  --csv path/to/your_data.csv \
  --text-col <column_name> \
  --output results.csv
```

Output columns added:

| Column | Description |
|--------|-------------|
| `label` | `Propaganda / Biased` or `Non-biased` |
| `confidence` | Model confidence 0–1 |
| `probability_biased` | Raw P(biased) — useful for sorting/ranking |
| `triggered_words` | Loaded/assertive words found in text |

### URL (scrape and classify)

```bash
pip install beautifulsoup4
python main.py --model-type transformer --url https://example.com/article
```

### Adjusting sensitivity

```bash
# More aggressive (higher recall, more false positives)
python main.py --model-type transformer --threshold 0.4 --text "..."

# More conservative (fewer false positives)
python main.py --model-type transformer --threshold 0.65 --text "..."
```

---

## Output explained

```
+-----------------------------------------------+
|  Label      : Propaganda / Biased             |
|  Confidence : [#############################-] |
|  P(biased)  : 0.9865                          |
+-----------------------------------------------+
|  Top linguistic signals:                      |
|    vader_compound         : -0.6219           |
|    loaded_word_ratio      :  0.2857           |
|    ...                                        |
|  Triggered words: corrupt, regime             |
+-----------------------------------------------+
|  Emotions detected:                           |
|  nrc: anger=0.1250  fear=0.1250  sadne=0.1250 |
|  hf : anger=0.9072  disgu=0.0468  sadne=0.0261|
+-----------------------------------------------+
```

### Emotion analysis

The `Emotions detected` section shows two complementary layers:

| Row | Method | What it does |
|-----|--------|--------------|
| `nrc:` | NRC Emotion Lexicon (word-level) | Looks up each word in a dictionary of 8 emotions (anger, fear, anticipation, trust, surprise, sadness, disgust, joy). Fast, always runs. Scores are normalised by word count. |
| `hf:` | DistilRoBERTa emotion model (sentence-level) | A pre-trained transformer that reads the full sentence in context. Returns probabilities across 7 emotions (anger, disgust, fear, joy, neutral, sadness, surprise). **Only runs when text is classified as propaganda** — not for non-biased text — to keep interactive mode fast. |

The top 3 emotions by score are shown for each method. Label abbreviations are truncated to 5 characters (e.g. `sadne` = sadness, `disgu` = disgust, `antic` = anticipation).

---

## Project structure

```
propaganda_detector/
├── data/
│   ├── raw/mbic.csv              # MBIC dataset (1,551 usable labeled sentences)
│   └── processed/                # Train/val/test splits
├── models/                       # Saved model weights (git-ignored)
├── src/
│   ├── data_loader.py            # MBIC dataset loading & splitting
│   ├── features.py               # NLP feature extraction (TF-IDF + linguistic)
│   ├── models.py                 # sklearn classifiers (SVM, RF, etc.)
│   ├── transformer_model.py      # DistilBERT fine-tuning & inference
│   ├── ptc_loader.py             # PTC v2 dataset loader (char-span → sentence labels)
│   ├── emotion_analyzer.py       # NRC + HuggingFace emotion analysis
│   └── predictor.py              # High-level predict API (PropagandaDetector / TransformerDetector)
├── train.py                      # Train sklearn models
├── train_transformer.py          # Fine-tune DistilBERT (supports --ptc-dir)
├── main.py                       # CLI entry point
├── api.py                        # FastAPI backend (uvicorn)
└── requirements.txt
```

---

## Dataset

**MBIC:** 1,551 sentences from 8 US news outlets (Breitbart, Fox News, Reuters, HuffPost, etc.) across 14 political topics, each labeled **Biased** or **Non-biased** by human annotators.

**PTC v2 (optional, not included):** ~15,700 sentences from 446 news articles annotated for 18 propaganda techniques at the [SemEval 2020 Task 11](https://propaganda.math.unipd.it/semeval2020task11/). Used only for training augmentation; not redistributed.
