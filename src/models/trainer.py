"""
src/models/trainer.py

Training entry point for the TruthLens deepfake detection MVP.

Wires together the already-implemented pipeline components:

    get_config()               -> src.config
    DatasetLoader               -> src.data.loader
    preprocessing pipeline      -> src.data.preprocessing
    MobileNetV2 model builder   -> src.models.mobilenetv2_model

and runs a single, simple model.fit() training run (frozen MobileNetV2
backbone, classification head only), saving the best model, training
history, and diagnostic plots under artifacts/.

Run from the project root with:

    python -m src.models.trainer
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving plots to disk
import matplotlib.pyplot as plt

from src.config import get_config
from src.data.loader import DatasetLoader
from src.data.preprocessing import Preprocessor
from src.models.mobilenetv2_model import MobileNetV2Classifier


logger = logging.getLogger("truthlens.trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds for reproducibility."""
    logger.info("Setting global random seed to %d", seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Dataset preparation
# --------------------------------------------------------------------------- #
def prepare_datasets(config: Any) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load and preprocess the train and validation datasets using the
    existing, already-tested DatasetLoader and Preprocessor components.

    Returns
    -------
    (train_ds, val_ds): a tuple of ready-to-fit tf.data.Dataset pipelines.
    """
    logger.info("Loading datasets via DatasetLoader")
    loader = DatasetLoader(config)

    train_ds = loader.load_train_dataset()
    val_ds = loader.load_validation_dataset()

    logger.info("Applying existing preprocessing pipeline (train: augmented, val: not)")
    preprocessor = Preprocessor(config)

    train_ds = preprocessor.prepare_train_dataset(train_ds)
    val_ds = preprocessor.prepare_eval_dataset(val_ds)

    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Model compilation
# --------------------------------------------------------------------------- #
def compile_model(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    """Compile the MobileNetV2 classification head for binary classification."""
    logger.info("Compiling model with learning_rate=%s", learning_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def build_callbacks(config: Any) -> list[tf.keras.callbacks.Callback]:
    """Build the training callbacks: checkpoint, early stopping, LR decay, CSV log."""
    checkpoint_path = Path(config.paths.checkpoint_dir) / "truthlens_mobilenetv2_best.keras"
    csv_log_path = Path(config.paths.log_dir) / "training_log.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path), append=False),
    ]

    logger.info("Checkpoints will be saved to: %s", checkpoint_path)
    logger.info("CSV training log will be saved to: %s", csv_log_path)

    return callbacks


# --------------------------------------------------------------------------- #
# Artifact saving
# --------------------------------------------------------------------------- #
def save_final_model(model: tf.keras.Model, config: Any) -> Path:
    """Save the final (best-weights-restored) model to artifacts/models/."""
    model_path = Path(config.paths.model_save_dir) / "truthlens_mobilenetv2.keras"
    model.save(model_path)
    logger.info("Final model saved to: %s", model_path)
    return model_path


def save_training_history(history: tf.keras.callbacks.History, config: Any) -> Path:
    """Save only the history keys that actually exist to a JSON file."""
    metrics_dir = Path(config.paths.artifacts_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history_path = metrics_dir / "training_history.json"

    history_dict: Dict[str, list] = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }

    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    logger.info("Training history saved to: %s", history_path)
    return history_path


def generate_plots(history: tf.keras.callbacks.History, config: Any) -> None:
    """Generate simple accuracy/loss/AUC plots from training history."""
    plots_dir = Path(config.paths.artifacts_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hist = history.history

    def _plot_metric(metric_key: str, val_key: str, title: str, filename: str) -> None:
        if metric_key not in hist:
            logger.warning("Metric '%s' not found in history, skipping plot.", metric_key)
            return

        plt.figure()
        plt.plot(hist[metric_key], label=f"train_{metric_key}")
        if val_key in hist:
            plt.plot(hist[val_key], label=val_key)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric_key)
        plt.legend()
        plt.tight_layout()

        out_path = plots_dir / filename
        plt.savefig(out_path)
        plt.close()
        logger.info("Saved plot: %s", out_path)

    _plot_metric("accuracy", "val_accuracy", "Training vs Validation Accuracy", "training_accuracy.png")
    _plot_metric("loss", "val_loss", "Training vs Validation Loss", "training_loss.png")
    _plot_metric("auc", "val_auc", "Training vs Validation AUC", "training_auc.png")


# --------------------------------------------------------------------------- #
# Main training pipeline
# --------------------------------------------------------------------------- #
def train() -> None:
    """Run the full TruthLens MVP training pipeline end to end."""
    logger.info("Starting TruthLens training pipeline")

    # 1. Configuration
    config = get_config()

    try:
        config.validate()
    except Exception as exc:
        logger.error("Configuration validation failed: %s", exc)
        raise

    try:
        config.ensure_output_dirs()
    except Exception as exc:
        logger.error("Failed to create output directories: %s", exc)
        raise

    set_global_seed(config.training.random_seed)

    # 2. Datasets
    try:
        train_ds, val_ds = prepare_datasets(config)
    except Exception as exc:
        logger.error("Failed to load or preprocess datasets: %s", exc)
        raise

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Model
    try:
        classifier = MobileNetV2Classifier(config)
        model = classifier.build()
    except Exception as exc:
        logger.error("Failed to build model: %s", exc)
        raise

    model = compile_model(model, learning_rate=config.training.learning_rate)
    model.summary(print_fn=logger.info)

    # 4. Callbacks
    callbacks = build_callbacks(config)

    # 5. Train
    logger.info("Beginning training for %d epochs", config.training.epochs)
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.training.epochs,
            callbacks=callbacks,
            verbose=1,
        )
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        raise

    # 6. Save artifacts
    model_path = save_final_model(model, config)
    save_training_history(history, config)
    generate_plots(history, config)

    # 7. Summary
    best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))
    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print("Model saved to:")
    print(f"  {model_path}")


if __name__ == "__main__":
    train()