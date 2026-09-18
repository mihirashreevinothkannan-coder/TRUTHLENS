"""
src/models/evaluator.py

Evaluation module for TruthLens.

Loads the already-trained MobileNetV2 deepfake-detection model and
evaluates it on the held-out TEST dataset only. Produces test metrics
(loss, accuracy, precision, recall, AUC), a confusion matrix plot, and
a JSON metrics report.

This module does NOT retrain the model and does NOT touch the
train/validation datasets.

Run from the project root with:

    python -m src.models.evaluator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from src.config import get_config, Config
from src.data.loader import DatasetLoader
from src.data.preprocessing import Preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Class ordering used throughout the project.
# 0 -> fake, 1 -> real (must match config.model.class_names)
CLASS_NAMES = ("fake", "real")
PREDICTION_THRESHOLD = 0.5


def load_test_model(config: Config) -> tf.keras.Model:
    """
    Load the already-trained TruthLens model from disk.

    Args:
        config: Frozen project configuration.

    Returns:
        The loaded Keras model.

    Raises:
        FileNotFoundError: If the trained model file does not exist.
    """
    model_path = Path(config.paths.model_save_dir) / "truthlens_mobilenetv2.keras"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Make sure the model has been trained and saved before running "
            "evaluation (see src/models/trainer.py)."
        )

    logger.info("Loading trained model from: %s", model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully.")
    return model


def evaluate_model(model: tf.keras.Model, test_ds: tf.data.Dataset) -> dict:
    """
    Run model.evaluate() on the preprocessed test dataset.

    Args:
        model: Loaded, trained Keras model.
        test_ds: Preprocessed, batched test tf.data.Dataset.

    Returns:
        Dictionary of metric name -> value.
    """
    logger.info("Evaluating model on test dataset...")
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    logger.info("Evaluation complete.")
    return results


def generate_predictions(
    model: tf.keras.Model, test_ds: tf.data.Dataset
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and true labels for the test dataset.

    Iterates the batched dataset once, collecting only the integer
    labels and thresholded binary predictions (memory-efficient,
    no need to hold raw images in memory).

    Args:
        model: Loaded, trained Keras model.
        test_ds: Preprocessed, batched test tf.data.Dataset (shuffle=False).

    Returns:
        Tuple of (true_labels, predicted_labels) as 1D numpy int arrays,
        using class ordering 0=fake, 1=real.
    """
    logger.info("Generating predictions for confusion matrix...")

    true_labels = []
    pred_labels = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        probs = np.asarray(probs).reshape(-1)

        # sigmoid output: >= threshold -> real (1), < threshold -> fake (0)
        batch_preds = (probs >= PREDICTION_THRESHOLD).astype(int)

        true_labels.append(labels.numpy().astype(int).reshape(-1))
        pred_labels.append(batch_preds)

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)

    logger.info("Generated predictions for %d test samples.", len(true_labels))
    return true_labels, pred_labels


def save_metrics(metrics: dict, output_path: Path) -> None:
    """
    Save evaluation metrics as a JSON file.

    Args:
        metrics: Dictionary of metric name -> value.
        output_path: Full path to the JSON file to write.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all values are JSON-serializable native Python types.
    clean_metrics = {k: float(v) for k, v in metrics.items()}

    with open(output_path, "w") as f:
        json.dump(clean_metrics, f, indent=4)

    logger.info("Saved test metrics to: %s", output_path)


def plot_confusion_matrix(
    true_labels: np.ndarray, pred_labels: np.ndarray, output_path: Path
) -> np.ndarray:
    """
    Compute and save a confusion matrix plot.

    Args:
        true_labels: Ground-truth integer labels (0=fake, 1=real).
        pred_labels: Predicted integer labels (0=fake, 1=real).
        output_path: Full path to save the PNG image.

    Returns:
        The raw confusion matrix as a numpy array, with rows=actual,
        columns=predicted, ordered [fake, real].
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title("TruthLens Test Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)

    # Annotate each cell with its count.
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    logger.info("Saved confusion matrix plot to: %s", output_path)
    return cm


def _print_summary(results: dict) -> None:
    """Print a clean, human-readable evaluation summary to the terminal."""
    loss = results.get("loss", float("nan"))
    accuracy = results.get("accuracy", float("nan"))
    precision = results.get("precision", float("nan"))
    recall = results.get("recall", float("nan"))
    auc = results.get("auc", float("nan"))

    print("=" * 50)
    print("TruthLens Test Evaluation")
    print("=" * 50)
    print(f"Test Loss      : {loss:.4f}")
    print(f"Test Accuracy  : {accuracy * 100:.2f}%")
    print(f"Test Precision : {precision * 100:.2f}%")
    print(f"Test Recall    : {recall * 100:.2f}%")
    print(f"Test AUC       : {auc:.4f}")
    print("=" * 50)


def _print_confusion_matrix(cm: np.ndarray) -> None:
    """Print the confusion matrix in a readable format to the terminal."""
    print("\nConfusion Matrix (rows=Actual, cols=Predicted)")
    print(f"{'':15s}{'Pred Fake':>12s}{'Pred Real':>12s}")
    print(f"{'Actual Fake':15s}{cm[0, 0]:>12d}{cm[0, 1]:>12d}")
    print(f"{'Actual Real':15s}{cm[1, 0]:>12d}{cm[1, 1]:>12d}")
    print()


def main() -> None:
    """
    Run the full TruthLens test-set evaluation pipeline:

    1. Load config
    2. Load test dataset only
    3. Preprocess test dataset (no augmentation)
    4. Load trained model
    5. Evaluate model -> loss/accuracy/precision/recall/AUC
    6. Generate predictions -> confusion matrix
    7. Save metrics JSON and confusion matrix plot
    8. Print summary
    """
    config = get_config()
    config.validate()

    logger.info("Loading test dataset...")
    loader = DatasetLoader(config)
    test_ds = loader.load_test_dataset()

    logger.info("Preparing test dataset (resize + normalize, no augmentation)...")
    preprocessor = Preprocessor(config)
    test_ds = preprocessor.prepare_eval_dataset(test_ds)

    model = load_test_model(config)

    results = evaluate_model(model, test_ds)
    _print_summary(results)

    true_labels, pred_labels = generate_predictions(model, test_ds)

    metrics_path = Path(config.paths.artifacts_dir) / "metrics" / "test_metrics.json"
    save_metrics(results, metrics_path)

    plot_path = Path(config.paths.artifacts_dir) / "plots" / "test_confusion_matrix.png"
    cm = plot_confusion_matrix(true_labels, pred_labels, plot_path)
    _print_confusion_matrix(cm)

    logger.info("Evaluation pipeline complete.")


if __name__ == "__main__":
    main()