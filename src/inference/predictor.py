"""
src/inference/predictor.py

Single-image inference for TruthLens.

Loads the already-trained MobileNetV2 deepfake-detection model once and
exposes a simple `Predictor.predict(image)` API for the Streamlit MVP.

This module does NOT retrain the model, does NOT modify the architecture,
and does NOT implement Grad-CAM (see src/inference/gradcam.py for that).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.config import get_config, Config

logger = logging.getLogger(__name__)

# Class ordering used throughout the project: 0 = fake, 1 = real
PREDICTION_THRESHOLD = 0.5

ImageInput = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class PredictionResult:
    """Clean, easy-to-use result of a single-image prediction."""

    label: str  # "fake" or "real"
    confidence: float  # confidence in the predicted label, as a percentage (0-100)
    probability_real: float  # raw sigmoid output (probability of class "real")


class Predictor:
    """
    Loads the trained TruthLens model once and runs single-image inference.

    Usage:
        predictor = Predictor()
        result = predictor.predict(image)  # PIL.Image or filesystem path
    """

    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize the predictor and load the trained model into memory.

        Args:
            config: Optional pre-built Config. If not provided, loads via
                get_config().

        Raises:
            FileNotFoundError: If the trained model file does not exist.
        """
        self.config = config if config is not None else get_config()
        self.image_size = self.config.model.image_size  # e.g. (224, 224)
        self.class_names = self.config.model.class_names  # ("fake", "real")

        self.model = self._load_model()

    def _load_model(self) -> tf.keras.Model:
        """
        Load the trained .keras model from artifacts/models/.

        Returns:
            The loaded Keras model (uncompiled use is fine for inference).

        Raises:
            FileNotFoundError: If the model file is missing.
        """
        model_path = (
            Path(self.config.paths.model_save_dir) / "truthlens_mobilenetv2.keras"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                "Train and save the model first (see src/models/trainer.py) "
                "before running inference."
            )

        logger.info("Loading trained model from: %s", model_path)
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully.")
        return model

    def _load_image(self, image: ImageInput) -> Image.Image:
        """
        Load an image from a PIL.Image or a filesystem path into a PIL Image.

        Args:
            image: A PIL.Image.Image, or a str/Path to an image file.

        Returns:
            A PIL Image object.

        Raises:
            TypeError: If `image` is not a supported type.
            FileNotFoundError: If a given path does not exist.
            ValueError: If the file exists but is not a valid/readable image.
        """
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                return Image.open(image_path)
            except UnidentifiedImageError as exc:
                raise ValueError(
                    f"File is not a valid or supported image: {image_path}"
                ) from exc

        raise TypeError(
            "predict() expects a PIL.Image.Image or a filesystem path "
            f"(str/Path), got: {type(image)}"
        )

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Convert a PIL image into a normalized, batched numpy array matching
        the training preprocessing pipeline:

            RGB -> resize 224x224 -> normalize to [-1, 1] -> batch dimension

        Args:
            image: A PIL Image.

        Returns:
            A numpy array of shape (1, H, W, 3), dtype float32, values in
            approximately [-1, 1].

        Raises:
            ValueError: If the image cannot be converted/resized.
        """
        try:
            rgb_image = image.convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not convert image to RGB: {exc}") from exc

        try:
            resized_image = rgb_image.resize(
                self.image_size, resample=Image.BILINEAR
            )
        except Exception as exc:
            raise ValueError(f"Could not resize image: {exc}") from exc

        image_array = np.asarray(resized_image, dtype=np.float32)

        # Match Preprocessor: layers.Rescaling(scale=1/127.5, offset=-1.0)
        normalized = (image_array / 127.5) - 1.0

        batched = np.expand_dims(normalized, axis=0)  # (1, H, W, 3)
        return batched

    def predict(self, image: ImageInput) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            image: A PIL.Image.Image, or a filesystem path (str/Path) to
                an image file.

        Returns:
            A PredictionResult with label, confidence (%), and the raw
            probability of the "real" class.

        Raises:
            TypeError: If `image` is not a supported input type.
            FileNotFoundError: If a given image path does not exist.
            ValueError: If the image is invalid, corrupt, or unreadable.
        """
        pil_image = self._load_image(image)
        batched_input = self._preprocess(pil_image)

        raw_output = self.model.predict(batched_input, verbose=0)
        probability_real = float(np.asarray(raw_output).reshape(-1)[0])

        if probability_real >= PREDICTION_THRESHOLD:
            label = "real"
            confidence = probability_real
        else:
            label = "fake"
            confidence = 1.0 - probability_real

        return PredictionResult(
            label=label,
            confidence=round(confidence * 100.0, 2),
            probability_real=round(probability_real, 4),
        )


def _run_manual_test() -> None:
    """
    Tiny standalone smoke test.

    Loads the model and runs prediction on one image path provided as a
    command-line argument. Does not modify any other project files.

    Usage:
        python -m src.inference.predictor path/to/image.jpg
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.predictor <path_to_image>")
        return

    image_path = sys.argv[1]

    predictor = Predictor()
    result = predictor.predict(image_path)

    print("=" * 40)
    print("TruthLens Single-Image Prediction")
    print("=" * 40)
    print(f"Predicted class : {result.label}")
    print(f"Confidence      : {result.confidence}%")
    print(f"P(real)         : {result.probability_real}")
    print("=" * 40)


if __name__ == "__main__":
    _run_manual_test()