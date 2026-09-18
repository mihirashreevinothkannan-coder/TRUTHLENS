"""Dataset loading module for the TruthLens deepfake detection project.

This module is responsible ONLY for loading raw images from disk into
`tf.data.Dataset` objects. It deliberately performs no preprocessing:

    - No pixel rescaling / normalization
    - No MobileNetV2-specific `preprocess_input` transformation
    - No data augmentation

Images are resized to a common shape during loading purely because
`tf.data.Dataset` batches require uniform tensor shapes -- this is a
structural necessity, not a preprocessing decision. All actual pixel-level
preprocessing belongs in `src/data/preprocessing.py`.

Typical usage:
    from src.config import get_config
    from src.data.loader import DatasetLoader

    config = get_config()
    loader = DatasetLoader(config)

    train_ds = loader.load_train_dataset()
    valid_ds = loader.load_validation_dataset()
    test_ds = loader.load_test_dataset()
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

from src.config import Config, get_config

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads TruthLens train/validation/test splits as `tf.data.Dataset` objects.

    This class encapsulates all interaction with `tf.keras.utils.image_dataset_from_directory`
    so that the (nearly identical) loading logic for each split lives in a single
    private method, avoiding duplicated code across train/valid/test loading.

    Attributes:
        config: The project's configuration object, providing paths, image
            size, batch size, class names, and random seed.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initializes the DatasetLoader.

        Args:
            config: A `Config` instance to source paths and hyperparameters
                from. If not provided, the default project configuration is
                loaded via `get_config()`. Injecting a config explicitly is
                useful for testing with alternate paths/settings.
        """
        self.config: Config = config if config is not None else get_config()
        logger.debug("DatasetLoader initialized with image_size=%s, batch_size=%s",
                     self.config.model.image_size, self.config.training.batch_size)

    def _load_dataset(
        self,
        directory: Path,
        split_name: str,
        shuffle: bool,
    ) -> tf.data.Dataset:
        """Loads a single dataset split from a directory of class subfolders.

        This is the shared core used by `load_train_dataset`,
        `load_validation_dataset`, and `load_test_dataset`. It centralizes:
            - Directory existence validation.
            - The call to `image_dataset_from_directory` with consistent
              arguments (image size, batch size, explicit class order).
            - AUTOTUNE prefetching.
            - Logging of dataset statistics.

        Args:
            directory: Path to the split directory (expected to contain
                one subfolder per class, e.g. `fake/` and `real/`).
            split_name: Human-readable name of the split (e.g. "train"),
                used only for logging and error messages.
            shuffle: Whether to shuffle the dataset. Should be True only
                for the training split; validation/test should remain
                unshuffled for deterministic, reproducible evaluation.

        Returns:
            A `tf.data.Dataset` yielding `(image_batch, label_batch)` pairs,
            with images resized to `config.model.image_size` and left in
            their original pixel value range (no rescaling applied), and
            with an `AUTOTUNE` prefetch applied.

        Raises:
            FileNotFoundError: If `directory` does not exist on disk.
        """
        if not directory.exists():
            logger.error("Dataset directory not found for split '%s': %s", split_name, directory)
            raise FileNotFoundError(
                f"Cannot load '{split_name}' split -- directory does not exist: "
                f"{directory}. Expected subfolders: "
                f"{self.config.model.class_names}."
            )

        logger.info(
            "Loading '%s' dataset from %s (image_size=%s, batch_size=%s, shuffle=%s)",
            split_name,
            directory,
            self.config.model.image_size,
            self.config.training.batch_size,
            shuffle,
        )

        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=directory,
            labels="inferred",
            label_mode="int",
            class_names=list(self.config.model.class_names),
            color_mode="rgb",
            batch_size=self.config.training.batch_size,
            image_size=self.config.model.image_size,
            shuffle=shuffle,
            seed=self.config.training.random_seed,
        )

        num_batches = tf.data.experimental.cardinality(dataset).numpy()
        logger.info(
            "Loaded '%s' dataset: %s batches of size %s (classes=%s)",
            split_name,
            num_batches,
            self.config.training.batch_size,
            self.config.model.class_names,
        )

        # Overlap data loading/preprocessing with model execution downstream.
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def load_train_dataset(self) -> tf.data.Dataset:
        """Loads the training split.

        The training dataset is shuffled (seeded for reproducibility) since
        shuffling during training helps prevent the model from learning
        spurious ordering patterns in the data.

        Returns:
            A `tf.data.Dataset` of `(image_batch, label_batch)` pairs for
            training, unpreprocessed and AUTOTUNE-prefetched.
        """
        return self._load_dataset(
            directory=self.config.paths.train_dir,
            split_name="train",
            shuffle=True,
        )

    def load_validation_dataset(self) -> tf.data.Dataset:
        """Loads the validation split.

        Shuffling is disabled so validation metrics are computed
        deterministically and are comparable across epochs/runs.

        Returns:
            A `tf.data.Dataset` of `(image_batch, label_batch)` pairs for
            validation, unpreprocessed and AUTOTUNE-prefetched.
        """
        return self._load_dataset(
            directory=self.config.paths.valid_dir,
            split_name="validation",
            shuffle=False,
        )

    def load_test_dataset(self) -> tf.data.Dataset:
        """Loads the test split.

        Shuffling is disabled so test predictions can be reliably matched
        back to their source files/labels for evaluation and reporting.

        Returns:
            A `tf.data.Dataset` of `(image_batch, label_batch)` pairs for
            testing, unpreprocessed and AUTOTUNE-prefetched.
        """
        return self._load_dataset(
            directory=self.config.paths.test_dir,
            split_name="test",
            shuffle=False,
        )

    def load_all_datasets(
        self,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Convenience method to load all three splits in one call.

        Returns:
            A tuple of `(train_dataset, validation_dataset, test_dataset)`.
        """
        train_ds = self.load_train_dataset()
        valid_ds = self.load_validation_dataset()
        test_ds = self.load_test_dataset()
        return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # Manual sanity check when running this module directly:
    #   python -m src.data.loader
    logging.basicConfig(level=logging.INFO)

    cfg = get_config()
    cfg.validate()  # Fail fast with a clear error if raw data is missing.

    dataset_loader = DatasetLoader(cfg)
    train_dataset, validation_dataset, test_dataset = dataset_loader.load_all_datasets()

    for images, labels in train_dataset.take(1):
        logger.info("Sample train batch -- images: %s, labels: %s", images.shape, labels.shape)