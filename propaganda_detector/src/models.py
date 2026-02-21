"""
ML model definitions, training, and evaluation for propaganda detection.

Supported classifiers
---------------------
  logistic    – Logistic Regression (fast baseline)
  svm         – Linear SVM (strong for text)
  rf          – Random Forest
  gradient    – Gradient Boosting (scikit-learn)
  voting      – Soft-voting ensemble of the above

All classifiers share a common interface via the ModelRegistry.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classifier zoo
# ---------------------------------------------------------------------------

def _logistic(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", class_weight="balanced", n_jobs=-1
    )


def _svm(C: float = 1.0) -> CalibratedClassifierCV:
    """LinearSVC wrapped in CalibratedClassifierCV to get probability estimates."""
    base = LinearSVC(C=C, class_weight="balanced", max_iter=2000)
    return CalibratedClassifierCV(base, cv=3)


def _random_forest(
    n_estimators: int = 300, max_depth: Optional[int] = None
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _gradient_boosting(
    n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 4
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )


def _voting_ensemble() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("logistic", _logistic()),
            ("svm", _svm()),
            ("rf", _random_forest(n_estimators=200)),
        ],
        voting="soft",
        n_jobs=-1,
    )


CLASSIFIER_REGISTRY: Dict[str, Any] = {
    "logistic": _logistic,
    "svm": _svm,
    "rf": _random_forest,
    "gradient": _gradient_boosting,
    "voting": _voting_ensemble,
}


# ---------------------------------------------------------------------------
# Full sklearn pipeline (features + classifier)
# ---------------------------------------------------------------------------

def build_pipeline(
    classifier_name: str = "logistic",
    max_tfidf_features: int = 15_000,
    **clf_kwargs,
) -> Pipeline:
    """
    Build an end-to-end sklearn Pipeline:
      raw text → features → classifier

    Parameters
    ----------
    classifier_name : one of the keys in CLASSIFIER_REGISTRY.
    max_tfidf_features : vocabulary size for TF-IDF.
    **clf_kwargs : keyword arguments forwarded to the classifier constructor.
    """
    if classifier_name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(CLASSIFIER_REGISTRY)}"
        )

    feature_union = build_feature_pipeline(max_tfidf_features)

    clf_factory = CLASSIFIER_REGISTRY[classifier_name]
    # voting ensemble takes no kwargs
    clf = clf_factory(**clf_kwargs) if classifier_name != "voting" else clf_factory()

    pipeline = Pipeline(
        steps=[
            ("features", feature_union),
            ("clf", clf),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    pipeline: Pipeline,
    train_texts: pd.Series,
    train_labels: pd.Series,
) -> Pipeline:
    """Fit the pipeline and return it."""
    logger.info("Training pipeline (%s) on %d samples…", pipeline.steps[-1][0], len(train_texts))
    t0 = time.perf_counter()
    pipeline.fit(train_texts.tolist(), train_labels.tolist())
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1fs", elapsed)
    return pipeline


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    pipeline: Pipeline,
    texts: pd.Series,
    labels: pd.Series,
    split_name: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate a trained pipeline and return a metrics dict.

    Returns
    -------
    Dict with keys: accuracy, f1, auc, report, confusion_matrix
    """
    preds = pipeline.predict(texts.tolist())
    proba = pipeline.predict_proba(texts.tolist())[:, 1]

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    auc = roc_auc_score(labels, proba)
    report = classification_report(
        labels, preds, target_names=["Non-biased", "Biased"]
    )
    cm = confusion_matrix(labels, preds)

    logger.info("=== %s results ===", split_name.upper())
    logger.info("Accuracy : %.4f", acc)
    logger.info("F1 score : %.4f", f1)
    logger.info("ROC-AUC  : %.4f", auc)
    logger.info("\n%s", report)

    return {
        "split": split_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
    }


def compare_models(
    train_texts: pd.Series,
    train_labels: pd.Series,
    val_texts: pd.Series,
    val_labels: pd.Series,
    classifiers: Optional[list] = None,
    max_tfidf_features: int = 15_000,
) -> Tuple[str, Pipeline, Dict]:
    """
    Train and evaluate multiple classifiers; return the best one by F1.

    Parameters
    ----------
    classifiers : list of classifier names to try.  Defaults to all.

    Returns
    -------
    (best_name, best_pipeline, results_dict)
    """
    if classifiers is None:
        classifiers = ["logistic", "svm", "rf", "gradient"]

    results = {}
    pipelines = {}

    for name in classifiers:
        logger.info("--- Fitting: %s ---", name)
        pipe = build_pipeline(name, max_tfidf_features=max_tfidf_features)
        pipe = train(pipe, train_texts, train_labels)
        metrics = evaluate(pipe, val_texts, val_labels, split_name=f"val/{name}")
        results[name] = metrics
        pipelines[name] = pipe

    # Pick best by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    logger.info("Best model: %s (val F1=%.4f)", best_name, results[best_name]["f1"])

    return best_name, pipelines[best_name], results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, name: str = "best_model") -> Path:
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Saved pipeline to %s", path)
    return path


def load_pipeline(name: str = "best_model") -> Pipeline:
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run train.py first."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded pipeline from %s", path)
    return pipeline
