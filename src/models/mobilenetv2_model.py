"""MobileNetV2 transfer-learning architecture for the TruthLens project.

This module is responsible ONLY for defining the model architecture. It
builds a binary (fake/real) classifier on top of a frozen, ImageNet-pretrained
MobileNetV2 feature extractor. It does not load data, preprocess images,
compile the model, or train it -- those responsibilities belong to
`src/data/loader.py`, `src/data/preprocessing.py`, and `src/models/trainer.py`
respectively.

Architecture:
    Input(H, W, 3)
      -> MobileNetV2 base (frozen, ImageNet weights, no top)
      -> GlobalAveragePooling2D
      -> Dropout
      -> Dense(dense_units, relu)
      -> Dense(1, sigmoid)                 [binary classification output]

Typical usage:
    from src.config import get_config
    from src.models.mobilenetv2_model import MobileNetV2Classifier

    config = get_config()
    classifier = MobileNetV2Classifier(config)
    model = classifier.build()
    classifier.summary()
"""

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from src.config import Config, get_config

logger = logging.getLogger(__name__)


class MobileNetV2Classifier:
    """Builds a MobileNetV2-based binary classifier via transfer learning.

    Attributes:
        config: The project's configuration object, providing input shape,
            MobileNetV2 parameters, and classification head settings.
        model: The built `tf.keras.Model`, populated after calling `build()`.
            `None` until `build()` is called.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initializes the classifier builder.

        Args:
            config: A `Config` instance providing model architecture
                settings. If not provided, the default project
                configuration is loaded via `get_config()`.
        """
        self.config: Config = config if config is not None else get_config()
        self.model: Optional[tf.keras.Model] = None

    def _build_base_model(self) -> tf.keras.Model:
        """Instantiates the pretrained MobileNetV2 feature extractor.

        The base is loaded with `include_top=False` to discard the original
        ImageNet 1000-class head, keeping only the convolutional feature
        extractor. Its `trainable` flag is set according to
        `config.model.freeze_base_layers` -- freezing prevents the
        pretrained weights from being updated while the new classification
        head is trained, which avoids destroying useful learned features on
        a comparatively small dataset.

        Returns:
            The MobileNetV2 base model (frozen or trainable per config).
        """
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.model.input_shape,
            alpha=self.config.model.mobilenet_alpha,
            include_top=False,
            weights=self.config.model.mobilenet_weights,
        )

        base_model.trainable = not self.config.model.freeze_base_layers

        logger.info(
            "MobileNetV2 base loaded (alpha=%s, weights=%s, input_shape=%s, "
            "trainable=%s)",
            self.config.model.mobilenet_alpha,
            self.config.model.mobilenet_weights,
            self.config.model.input_shape,
            base_model.trainable,
        )

        return base_model

    def build(self) -> tf.keras.Model:
        """Builds and returns the full classification model.

        Constructs the Keras Functional API graph: input -> MobileNetV2 base
        (called with `training=False` to keep BatchNormalization layers in
        inference mode, regardless of whether the base is frozen) ->
        GlobalAveragePooling2D -> Dropout -> Dense(relu) -> Dense(sigmoid).

        Returns:
            The compiled-free (architecture only) `tf.keras.Model`, ready to
            be compiled and trained by `src/models/trainer.py`.
        """
        base_model = self._build_base_model()

        inputs = tf.keras.Input(shape=self.config.model.input_shape, name="input_image")

        # training=False keeps BatchNorm layers in inference mode during
        # model.fit(), which is required for correct behavior with a frozen
        # base and protects against surprises if the base is unfrozen later.
        x = base_model(inputs, training=False)

        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
        x = layers.Dropout(rate=self.config.model.dropout_rate, name="head_dropout")(x)
        x = layers.Dense(
            units=self.config.model.dense_units,
            activation="relu",
            name="head_dense",
        )(x)
        outputs = layers.Dense(units=1, activation="sigmoid", name="predictions")(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="truthlens_mobilenetv2")

        logger.info(
            "Built model '%s' with %d total parameters (%d trainable)",
            self.model.name,
            self.model.count_params(),
            sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights),
        )

        return self.model

    def summary(self) -> None:
        """Logs the model's architecture summary.

        Raises:
            RuntimeError: If called before `build()`.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been built yet. Call `build()` before `summary()`."
            )
        self.model.summary(print_fn=logger.info)


if __name__ == "__main__":
    # Manual sanity check when running this module directly:
    #   python -m src.models.mobilenetv2_model
    logging.basicConfig(level=logging.INFO)

    cfg = get_config()
    classifier = MobileNetV2Classifier(cfg)
    built_model = classifier.build()
    classifier.summary()

    # Quick forward-pass shape check with random data (no dataset loading).
    dummy_input = tf.random.uniform(shape=(2, *cfg.model.input_shape))
    dummy_output = built_model(dummy_input, training=False)
    logger.info("Sample forward pass -- output shape: %s", dummy_output.shape)