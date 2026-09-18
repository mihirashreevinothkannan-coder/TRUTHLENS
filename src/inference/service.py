"""Structured inference adapter separating TruthLens UI from model execution."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Optional
from PIL import Image
from src.inference.predictor import Predictor
from src.inference.gradcam import GradCAM

@dataclass(frozen=True)
class ImageMetadata:
    width: int; height: int; format: str; mode: str
    @property
    def aspect_ratio(self) -> str: return f"{self.width / self.height:.2f}:1" if self.height else "Unknown"

@dataclass(frozen=True)
class AnalysisResponse:
    prediction: str; real_probability: float; fake_probability: float; confidence: float
    model: str; inference_mode: str; gradcam_available: bool; gradcam_image: Optional[str]
    metadata: ImageMetadata; processing_time_ms: int; analyzed_at: str; threshold: float = .5
    gradcam_heatmap: Optional[bytes] = None
    gradcam_overlay: Optional[bytes] = None
    target_class: Optional[str] = None
    target_layer: Optional[str] = None
    gradcam_error: Optional[str] = None
    def to_dict(self) -> dict[str, Any]: return asdict(self)

class AnalysisService:
    """Local adapter; replace here to use an API without changing the UI."""
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor
        self.gradcam = GradCAM(predictor.model)
    def analyze(self, image: Image.Image) -> AnalysisResponse:
        started = perf_counter()
        gradcam_result = None
        gradcam_error = None
        try:
            # One real forward/gradient pass produces the classification and
            # attention map from the same loaded weights and preprocessed input.
            gradcam_result = self.gradcam.generate(self.predictor._preprocess(image), image)
            real = gradcam_result.probability_real
            label = gradcam_result.target_class
            confidence = max(real, 1-real) * 100
        except Exception as exc:
            gradcam_error = f"{type(exc).__name__}: {exc}"
            # Keep classification available even when explainability itself fails.
            result = self.predictor.predict(image)
            real = float(result.probability_real)
            label = result.label
            confidence = float(result.confidence)
        return AnalysisResponse(label, real, round(1-real, 4), confidence,
            "TruthLens MobileNetV2", "REAL MODEL", gradcam_result is not None, None,
            ImageMetadata(image.width, image.height, image.format or "Unknown", image.mode),
            round((perf_counter()-started)*1000), datetime.now().astimezone().strftime("%d %b %Y, %I:%M:%S %p %Z"),
            gradcam_heatmap=gradcam_result.heatmap_png if gradcam_result else None,
            gradcam_overlay=gradcam_result.overlay_png if gradcam_result else None,
            target_class=gradcam_result.target_class if gradcam_result else None,
            target_layer=gradcam_result.target_layer if gradcam_result else None,
            gradcam_error=gradcam_error)
