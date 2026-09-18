"""Real Grad-CAM generation for the loaded TruthLens Keras model."""
from __future__ import annotations
from dataclasses import dataclass
import io
import numpy as np
import tensorflow as tf
from PIL import Image

@dataclass(frozen=True)
class GradCAMResult:
    heatmap_png: bytes; overlay_png: bytes; target_class: str; target_layer: str; probability_real: float

def _png(image: Image.Image) -> bytes:
    buffer=io.BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()

def _colorize(values: np.ndarray) -> Image.Image:
    """Map a normalized genuine Grad-CAM response to a readable RGB image."""
    x=np.clip(values,0.,1.);r=np.clip(1.5-np.abs(4*x-3),0,1);g=np.clip(1.5-np.abs(4*x-2),0,1);b=np.clip(1.5-np.abs(4*x-1),0,1)
    return Image.fromarray((np.stack((r,g,b),axis=-1)*255).astype(np.uint8),'RGB')

def create_overlay(original: Image.Image, heatmap_png: bytes, opacity: float=.45) -> bytes:
    """Blend an existing genuine Grad-CAM heatmap with its source image."""
    if not 0<=opacity<=1: raise ValueError('Grad-CAM overlay opacity must be between 0 and 1.')
    base=original.convert('RGB');heatmap=Image.open(io.BytesIO(heatmap_png)).convert('RGB').resize(base.size,Image.Resampling.BILINEAR)
    return _png(Image.blend(base,heatmap,opacity))

class GradCAM:
    """Compute Grad-CAM from a real loaded model and preprocessed image."""
    def __init__(self, model: tf.keras.Model, backbone_name: str='mobilenetv2_1.00_224', target_layer_name: str='out_relu') -> None:
        self.model=model;self.backbone_name=backbone_name;self.target_layer_name=target_layer_name
        backbone=model.get_layer(backbone_name);target=backbone.get_layer(target_layer_name)
        # The target is nested inside MobileNetV2, so expose it together with
        # the backbone output in one call. This preserves the real path from
        # target activation through the trained classification head.
        backbone_with_target=tf.keras.Model(backbone.input,[target.output,backbone.output])
        inputs=model.inputs[0];activation,features=backbone_with_target(inputs)
        x=model.get_layer('global_average_pooling')(features)
        x=model.get_layer('head_dropout')(x,training=False)
        x=model.get_layer('head_dense')(x)
        prediction=model.get_layer('predictions')(x)
        self.gradient_model=tf.keras.Model(inputs,[activation,prediction])
    def generate(self, preprocessed: np.ndarray, original: Image.Image) -> GradCAMResult:
        inputs=tf.convert_to_tensor(preprocessed,dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv,prediction=self.gradient_model(inputs,training=False)
            probability_real=float(prediction[0,0].numpy())
            target_class='real' if probability_real>=.5 else 'fake'
            score=prediction[:,0] if target_class=='real' else 1.0-prediction[:,0]
        gradients=tape.gradient(score,conv)
        if gradients is None: raise RuntimeError('No gradient was produced for the selected Grad-CAM layer.')
        weights=tf.reduce_mean(gradients,axis=(0,1,2));heatmap=tf.maximum(tf.reduce_sum(conv[0]*weights,axis=-1),0);maximum=float(tf.reduce_max(heatmap).numpy())
        if maximum<=0: raise RuntimeError('Grad-CAM heatmap has no positive activation for this prediction.')
        normalized=(heatmap/maximum).numpy();resized=np.asarray(Image.fromarray((normalized*255).astype(np.uint8)).resize(original.size,Image.Resampling.BILINEAR),dtype=np.float32)/255.
        heatmap_png=_png(_colorize(resized));return GradCAMResult(heatmap_png,create_overlay(original,heatmap_png),target_class,f'{self.backbone_name}/{self.target_layer_name}',probability_real)
