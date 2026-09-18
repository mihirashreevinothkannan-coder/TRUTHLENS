"""Preprocessing module for the TruthLens deepfake detection project.

This module is responsible ONLY for transforming already-loaded images:
resizing, normalization, and data augmentation. It has no knowledge of
file paths or directory structures -- loading raw images from disk is the
sole responsibility of `src/data/loader.py`.

Two independent Keras preprocessing pipelines are built:
    - `resize_and_rescale`: Resizing + normalization to [-1, 1]. Applied to
      EVERY split (train, validation, test) and eventually inference, since
      every image reaching the model must be in this shape/range.
    - `data_augmentation`: RandomFlip, RandomRotation, RandomZoom, and
      RandomContrast. Applied ONLY to the training split.

Typical usage:
    from src.config import get_config
    from src.data.loader import DatasetLoader
    from src.data.preprocessing import Preprocessor

    config = get_config()
    loader = DatasetLoader(config)
    preprocessor = Preprocessor(config)

    train_ds = preprocessor.prepare_train_dataset(loader.load_train_dataset())
    valid_ds = preprocessor.prepare_eval_dataset(loader.load_validation_dataset())
    test_ds = preprocessor.prepare_eval_dataset(loader.load_test_dataset())
"""

import logging
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from src.config import Config, get_config

logger = logging.getLogger(__name__)


class Preprocessor:
    """Builds and applies resize/normalize and augmentation pipelines.

    Attributes:
        config: The project's configuration object, providing the target
            image size used for resizing.
        rotation_factor: Fraction of a full rotation (2*pi) used as the
            random rotation range, e.g. 0.1 -> images rotated randomly
            within roughly +/- 36 degrees.
        zoom_factor: Fraction used as the random zoom range for both height
            and width, e.g. 0.1 -> zoom randomly within +/- 10%.
        contrast_factor: Fraction used as the random contrast adjustment
            range, e.g. 0.1 -> contrast randomly adjusted within +/- 10%.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rotation_factor: float = 0.1,
        zoom_factor: float = 0.1,
        contrast_factor: float = 0.1,
    ) -> None:
        """Initializes the Preprocessor and builds both pipelines up front.

        Args:
            config: A `Config` instance providing the target image size.
                If not provided, the default project configuration is
                loaded via `get_config()`.
            rotation_factor: Random rotation intensity (see class docstring).
            zoom_factor: Random zoom intensity (see class docstring).
            contrast_factor: Random contrast intensity (see class docstring).
        """
        self.config: Config = config if config is not None else get_config()
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.contrast_factor = contrast_factor

        self._resize_and_rescale = self._build_resize_rescale_pipeline()
        self._augmentation = self._build_augmentation_pipeline()

        logger.debug(
            "Preprocessor initialized: image_size=%s, rotation_factor=%s, "
            "zoom_factor=%s, contrast_factor=%s",
            self.config.model.image_size,
            rotation_factor,
            zoom_factor,
            contrast_factor,
        )

    def _build_resize_rescale_pipeline(self) -> tf.keras.Sequential:
        """Builds the resize + normalization pipeline.

        Resizing guarantees every image matches the model's expected input
        shape regardless of its original dimensions. Rescaling maps pixel
        values from the raw `[0, 255]` range to `[-1, 1]` using
        `scale=1/127.5, offset=-1.0` -- this matches the exact normalization
        convention MobileNetV2's ImageNet-pretrained weights were trained
        with (Keras' `mobilenet_v2.preprocess_input`, "tf" mode). Using a
        different range here would silently hurt transfer-learning
        performance since the pretrained backbone's batch-norm statistics
        are calibrated to this input distribution.

        Returns:
            A `tf.keras.Sequential` model containing a `Resizing` layer
            followed by a `Rescaling` layer.
        """
        height, width = self.config.model.image_size
        pipeline = tf.keras.Sequential(
            [
                layers.Resizing(height, width, name="resize"),
                layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="rescale"),
            ],
            name="resize_and_rescale",
        )
        return pipeline

    def _build_augmentation_pipeline(self) -> tf.keras.Sequential:
        """Builds the data augmentation pipeline (training data only).

        Augmentation order matters somewhat but is not critical here: flip
        and rotation are applied first (geometric transforms), followed by
        zoom (also geometric), and finally contrast (a photometric
        transform). This pipeline is intentionally decoupled from
        `resize_and_rescale` so it can be reused, tested, or attached
        elsewhere (e.g., embedded directly in a model for serving) without
        touching resize/normalize logic.

        Returns:
            A `tf.keras.Sequential` model containing RandomFlip,
            RandomRotation, RandomZoom, and RandomContrast layers.
        """
        pipeline = tf.keras.Sequential(
            [
                layers.RandomFlip(mode="horizontal", name="random_flip"),
                layers.RandomRotation(factor=self.rotation_factor, name="random_rotation"),
                layers.RandomZoom(
                    height_factor=self.zoom_factor,
                    width_factor=self.zoom_factor,
                    name="random_zoom",
                ),
                layers.RandomContrast(factor=self.contrast_factor, name="random_contrast"),
            ],
            name="data_augmentation",
        )
        return pipeline

    def apply(self, dataset: tf.data.Dataset, augment: bool) -> tf.data.Dataset:
        """Applies resize/normalize (and optionally augmentation) to a dataset.

        Args:
            dataset: A `tf.data.Dataset` yielding `(image_batch, label_batch)`
                pairs, typically produced by `DatasetLoader`.
            augment: Whether to apply the augmentation pipeline after
                resizing/rescaling. This should be `True` only for the
                training split -- passing `True` for validation/test data
                would introduce random noise into evaluation metrics.

        Returns:
            A transformed `tf.data.Dataset` with images resized to
            `config.model.image_size`, normalized to `[-1, 1]`, and
            (if `augment=True`) randomly augmented. The dataset is
            re-prefetched with `AUTOTUNE` as the final step.
        """
        logger.info(
            "Applying resize+rescale%s to dataset",
            " and augmentation" if augment else "",
        )

        def _preprocess(
            images: tf.Tensor, labels: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            """Per-batch transformation function used inside `dataset.map()`."""
            images = self._resize_and_rescale(images, training=False)
            if augment:
                # training=True ensures randomness is actually applied
                # (these layers no-op under training=False).
                images = self._augmentation(images, training=True)
            return images, labels

        dataset = dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def prepare_train_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Prepares the training dataset: resize, rescale, and augment.

        Args:
            dataset: The raw training `tf.data.Dataset` from `DatasetLoader`.

        Returns:
            A fully preprocessed, augmented training dataset ready for
            `model.fit()`.
        """
        return self.apply(dataset, augment=True)

    def prepare_eval_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Prepares a validation or test dataset: resize and rescale only.

        No augmentation is applied here -- evaluation and test metrics must
        reflect the model's performance on unmodified (aside from mandatory
        resize/normalize) data.

        Args:
            dataset: The raw validation or test `tf.data.Dataset` from
                `DatasetLoader`.

        Returns:
            A preprocessed (but not augmented) dataset ready for
            `model.evaluate()` or `model.predict()`.
        """
        return self.apply(dataset, augment=False)


if __name__ == "__main__":
    # Manual sanity check when running this module directly:
    #   python -m src.data.preprocessing
    # Builds a small synthetic dataset in memory (no disk I/O) purely to
    # verify the pipelines run and produce correctly shaped/ranged output.
    logging.basicConfig(level=logging.INFO)

    cfg = get_config()
    preprocessor = Preprocessor(cfg)

    dummy_images = tf.random.uniform(
        shape=(4, 256, 256, 3), minval=0, maxval=255, dtype=tf.float32
    )
    dummy_labels = tf.constant([0, 1, 0, 1])
    dummy_dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(4)

    train_like_ds = preprocessor.prepare_train_dataset(dummy_dataset)
    eval_like_ds = preprocessor.prepare_eval_dataset(dummy_dataset)

    for imgs, lbls in train_like_ds.take(1):
        logger.info(
            "Train-prepared batch -- shape=%s, min=%.3f, max=%.3f",
            imgs.shape, float(tf.reduce_min(imgs)), float(tf.reduce_max(imgs)),
        )

    for imgs, lbls in eval_like_ds.take(1):
        logger.info(
            "Eval-prepared batch -- shape=%s, min=%.3f, max=%.3f",
            imgs.shape, float(tf.reduce_min(imgs)), float(tf.reduce_max(imgs)),
        )