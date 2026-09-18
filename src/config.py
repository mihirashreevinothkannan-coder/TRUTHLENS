"""Central configuration module for the TruthLens deepfake detection project.

This module is the single source of truth for all paths, hyperparameters,
and model settings used across the project. No other module should hardcode
a path, image size, batch size, or class label mapping -- everything must be
imported from here.

Design notes:
    - All configuration is expressed as frozen (immutable) dataclasses so
      that once a `Config` object is created, its values cannot be silently
      mutated by some other part of the pipeline.
    - Paths are built with `pathlib.Path`, anchored at `PROJECT_ROOT`, so the
      project runs correctly regardless of the machine or working directory
      it is launched from.
    - This module has zero heavy dependencies (no TensorFlow, no OpenCV) so
      it can be imported cheaply anywhere, including in lightweight tests.
    - Validation of the filesystem (e.g., "does data/raw/train/fake exist?")
      is NOT performed automatically on import. Call `Config.validate()`
      explicitly (typically from `main.py`) when you need that guarantee.

Typical usage:
    from src.config import get_config

    config = get_config()
    config.validate()  # optional, fails fast if raw data is missing
    print(config.model.image_size)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Project root: anchor point for every other path in this file.
# config.py lives at TruthLens/src/config.py, so parent.parent -> TruthLens/
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths used throughout the project.

    All paths are derived from `PROJECT_ROOT` so the project remains
    portable across machines and working directories.

    Attributes:
        project_root: Absolute path to the TruthLens project root.
        data_root: Root directory containing all dataset splits.
        train_dir: Directory containing training images (fake/real subfolders).
        valid_dir: Directory containing validation images (fake/real subfolders).
        test_dir: Directory containing test images (fake/real subfolders).
        artifacts_dir: Root directory for all generated artifacts (models, logs, etc.).
        model_save_dir: Directory where trained model files are saved.
        checkpoint_dir: Directory where training checkpoints are saved.
        log_dir: Directory where log files are written.
        gradcam_output_dir: Directory where Grad-CAM visualizations are saved.
    """

    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "data"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    valid_dir: Path = PROJECT_ROOT / "data" / "valid"
    test_dir: Path = PROJECT_ROOT / "data" / "test"

    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    model_save_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    checkpoint_dir: Path = PROJECT_ROOT / "artifacts" / "checkpoints"
    log_dir: Path = PROJECT_ROOT / "artifacts" / "logs"
    gradcam_output_dir: Path = PROJECT_ROOT / "artifacts" / "gradcam"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture settings, including MobileNetV2 transfer-learning params.

    Attributes:
        image_size: Target (height, width) that all input images are resized to.
        num_channels: Number of color channels in the input images (RGB = 3).
        num_classes: Number of output classes for the classifier.
        class_names: Ordered class labels. The index of a name in this tuple
            IS its integer label (e.g., "fake" -> 0, "real" -> 1). Every
            module that maps predictions to labels must use this tuple as
            the single source of truth.
        mobilenet_alpha: Width multiplier for MobileNetV2 (controls model size/speed).
        mobilenet_weights: Pretrained weight source (e.g., "imagenet" or None).
        freeze_base_layers: Whether the pretrained MobileNetV2 base is frozen
            during initial training (standard transfer-learning practice).
        dropout_rate: Dropout rate applied in the classification head.
        dense_units: Number of units in the dense layer of the classification head.
    """

    image_size: Tuple[int, int] = (224, 224)
    num_channels: int = 3
    num_classes: int = 2
    class_names: Tuple[str, ...] = ("fake", "real")

    mobilenet_alpha: float = 1.0
    mobilenet_weights: str = "imagenet"
    freeze_base_layers: bool = True
    dropout_rate: float = 0.3
    dense_units: int = 128

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Full input tensor shape expected by the model.

        Returns:
            A tuple of (height, width, channels), e.g., (224, 224, 3).
        """
        height, width = self.image_size
        return (height, width, self.num_channels)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters controlling the training process.

    Attributes:
        batch_size: Number of samples per training batch.
        epochs: Number of full passes over the training dataset.
        learning_rate: Initial learning rate for the optimizer.
        random_seed: Seed used for reproducibility across numpy/tensorflow/random.
        validation_split: Fraction of data reserved for validation, used only
            if a separate validation directory is not provided/found.
        early_stopping_patience: Number of epochs with no improvement before
            training is stopped early.
        shuffle_buffer_size: Buffer size used when shuffling the training dataset.
    """

    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-4
    random_seed: int = 42
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    shuffle_buffer_size: int = 1024


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundling all sub-configs.

    This is the single object that every other module in TruthLens should
    import and use. It intentionally has no logic beyond aggregation and
    validation, so it stays trivial to reason about and test.

    Attributes:
        paths: All filesystem path settings.
        model: All model architecture settings.
        training: All training hyperparameters.
    """

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        """Validate that required raw dataset directories exist on disk.

        This is an explicit, opt-in check (not run automatically on import
        or construction) so that config objects remain cheap to create in
        contexts like unit tests, where the raw dataset may not be present.

        Raises:
            FileNotFoundError: If any required raw data directory is missing.
        """
        required_dirs = [
            self.paths.train_dir / "fake",
            self.paths.train_dir / "real",
            self.paths.valid_dir / "fake",
            self.paths.valid_dir / "real",
            self.paths.test_dir / "fake",
            self.paths.test_dir / "real",
        ]

        missing_dirs = [str(d) for d in required_dirs if not d.exists()]

        if missing_dirs:
            missing_list = "\n  - ".join(missing_dirs)
            raise FileNotFoundError(
                "TruthLens config validation failed. The following required "
                f"dataset directories are missing:\n  - {missing_list}\n"
                "Please ensure the dataset is placed under "
                f"'{self.paths.data_root}' following the documented "
                "train/valid/test -> fake/real structure."
            )

    def ensure_output_dirs(self) -> None:
        """Create all output/artifact directories if they do not already exist.

        This covers directories TruthLens writes to (models, checkpoints,
        logs, Grad-CAM outputs) as opposed to `validate()`, which only reads
        from the raw dataset directories.
        """
        output_dirs = (
            self.paths.artifacts_dir,
            self.paths.model_save_dir,
            self.paths.checkpoint_dir,
            self.paths.log_dir,
            self.paths.gradcam_output_dir,
        )
        for directory in output_dirs:
            directory.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Factory function returning the project's configuration object.

    Using a factory function (rather than a bare module-level singleton)
    keeps construction explicit at call sites and makes it trivial to
    substitute a different config in tests, e.g.:

        test_config = Config(paths=PathConfig(data_root=tmp_path))

    Returns:
        A fully populated, immutable `Config` instance using default values.
    """
    return Config()


if __name__ == "__main__":
    # Simple manual sanity check when running this file directly:
    #   python -m src.config
    cfg = get_config()
    print(f"Project root: {cfg.paths.project_root}")
    print(f"Input shape:  {cfg.model.input_shape}")
    print(f"Class names:  {cfg.model.class_names}")
    print(f"Batch size:   {cfg.training.batch_size}")