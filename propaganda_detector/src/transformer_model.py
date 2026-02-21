"""
DistilBERT fine-tuning for propaganda / media-bias detection.

Uses HuggingFace `transformers` + PyTorch.

Architecture
------------
  distilbert-base-uncased  →  [CLS] pooling  →  Linear(768, 2)  →  softmax

Key design choices
------------------
- Class weights in the loss function to handle the 65/35 imbalance
- Linear LR warmup + decay (standard for fine-tuning)
- Max sequence length 128 (sentences rarely exceed 80 tokens)
- Gradient clipping at 1.0 to stabilise training
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BiasDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Trainer / wrapper
# ---------------------------------------------------------------------------

class DistilBertTrainer:
    """
    Fine-tunes distilbert-base-uncased for binary bias classification.

    After training, acts as a predictor with predict() / predict_proba().
    """

    def __init__(
        self,
        pretrained: str = PRETRAINED,
        max_len: int = 128,
        batch_size: int = 16,
        epochs: int = 4,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        device: Optional[str] = None,
    ):
        self.pretrained = pretrained
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.warmup_ratio = warmup_ratio
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> "DistilBertTrainer":
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
            get_linear_schedule_with_warmup,
        )

        logger.info("Loading tokenizer & model: %s", self.pretrained)
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self.pretrained)
        self._model = DistilBertForSequenceClassification.from_pretrained(
            self.pretrained, num_labels=2
        ).to(self.device)

        train_ds = BiasDataset(train_texts, train_labels, self._tokenizer, self.max_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Class weights to counteract 65/35 imbalance
        counts = np.bincount(train_labels)
        weights = torch.tensor(
            len(train_labels) / (2.0 * counts), dtype=torch.float
        ).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        logger.info(
            "Training on %s | epochs=%d | batch=%d | lr=%.0e | device=%s",
            self.pretrained, self.epochs, self.batch_size, self.lr, self.device,
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, loss_fn)
            msg = f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f}"

            if val_texts is not None:
                val_metrics = self._evaluate(val_texts, val_labels)
                msg += (
                    f" | val_acc={val_metrics['accuracy']:.4f}"
                    f" | val_f1={val_metrics['f1']:.4f}"
                )
            logger.info(msg)
            print(msg)

        return self

    def _train_epoch(self, loader, optimizer, scheduler, loss_fn) -> float:
        self._model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="  training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns shape (n, 2) probabilities: [:, 0]=non-biased, [:, 1]=biased."""
        self._model.eval()
        all_probs = []

        ds = BiasDataset(texts, [0] * len(texts), self._tokenizer, self.max_len)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self, texts: List[str], labels: List[int]) -> dict:
        from sklearn.metrics import accuracy_score, f1_score
        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="binary"),
        }

    def evaluate(
        self, texts: List[str], labels: List[int], split_name: str = "test"
    ) -> dict:
        from sklearn.metrics import (
            accuracy_score, classification_report, f1_score, roc_auc_score
        )
        proba = self.predict_proba(texts)
        preds = proba.argmax(axis=1)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary")
        auc = roc_auc_score(labels, proba[:, 1])
        report = classification_report(labels, preds, target_names=["Non-biased", "Biased"])

        logger.info("=== %s results ===", split_name.upper())
        logger.info("Accuracy : %.4f", acc)
        logger.info("F1       : %.4f", f1)
        logger.info("AUC      : %.4f", auc)
        logger.info("\n%s", report)
        print(f"\n=== {split_name.upper()} RESULTS ===")
        print(f"Accuracy : {acc:.4f}  |  F1 : {f1:.4f}  |  AUC : {auc:.4f}")
        print(report)

        return {"accuracy": acc, "f1": f1, "auc": auc, "report": report}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, name: str = "distilbert_model") -> Path:
        out_dir = MODEL_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(out_dir)
        self._tokenizer.save_pretrained(out_dir)
        # Save hyperparams so we can reconstruct the object
        import json
        meta = {
            "pretrained": self.pretrained,
            "max_len": self.max_len,
            "batch_size": self.batch_size,
        }
        (out_dir / "trainer_meta.json").write_text(json.dumps(meta))
        logger.info("Saved transformer model to %s", out_dir)
        print(f"Model saved to: {out_dir}")
        return out_dir

    @classmethod
    def load(cls, name: str = "distilbert_model") -> "DistilBertTrainer":
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )
        import json

        out_dir = MODEL_DIR / name
        if not out_dir.exists():
            raise FileNotFoundError(
                f"No transformer model at {out_dir}.\n"
                "Train one with: python train_transformer.py --data data/raw/mbic.csv"
            )

        meta = json.loads((out_dir / "trainer_meta.json").read_text())
        trainer = cls(
            pretrained=meta["pretrained"],
            max_len=meta["max_len"],
            batch_size=meta["batch_size"],
        )
        trainer._tokenizer = DistilBertTokenizerFast.from_pretrained(str(out_dir))
        trainer._model = DistilBertForSequenceClassification.from_pretrained(
            str(out_dir)
        ).to(trainer.device)
        trainer._model.eval()
        logger.info("Loaded transformer model from %s", out_dir)
        return trainer
