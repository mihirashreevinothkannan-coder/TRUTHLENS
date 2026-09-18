TruthLens — Current Project Analysis
TruthLens is now a working local AI image-authenticity application built around a trained MobileNetV2 binary classifier.
What it does
Users can upload a JPG, PNG, or WebP image and receive:
- Model prediction: LIKELY REAL or LIKELY FAKE
- Real and fake probabilities
- Model confidence
- Input image metadata
- Processing time
- Real Grad-CAM heatmap
- Real Grad-CAM overlay with adjustable opacity
- Analysis history for the current browser session
- Research and responsible-AI guidance
The application correctly describes results as model predictions, not definitive proof.
Current architecture
Streamlit UI
    ↓
AnalysisService
    ↓
Predictor / Loaded Keras model
    ↓
MobileNetV2 backbone + binary classifier head
    ↓
Prediction + real Grad-CAM
    ↓
Structured AnalysisResponse
    ↓
Results dashboard
Important files
File	Responsibility
[web/app.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\web\app.py)	Streamlit UI, upload flow, results, history, research page
[src/inference/service.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\inference\service.py)	Clean frontend-facing analysis adapter and structured response
[src/inference/predictor.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\inference\predictor.py)	Model loading and existing image preprocessing
[src/inference/gradcam.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\inference\gradcam.py)	Genuine Grad-CAM generation and overlay blending
[src/models/mobilenetv2_model.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\models\mobilenetv2_model.py)	MobileNetV2 transfer-learning architecture
[src/data/preprocessing.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\data\preprocessing.py)	Training/evaluation preprocessing
[src/config.py](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\src\config.py)	Paths, model settings, classes, training parameters
[artifacts/models/truthlens_mobilenetv2.keras](C:\Users\MIHIRA SHREE\Desktop\FirstReact\TRUTHLENS\artifacts\models\truthlens_mobilenetv2.keras)	Trained real model artifact


Model details
- Architecture: MobileNetV2 with transfer learning
- Input: 224 × 224 × 3
- Classes:
  - fake = 0
  - real = 1
- Output: sigmoid probability representing P(real)
- Threshold: 0.50
- Classification head:
  - GlobalAveragePooling2D
  - Dropout 0.30
  - Dense ReLU, 128 units
  - Sigmoid output
Grad-CAM details
The real Grad-CAM implementation uses:
Backbone: mobilenetv2_1.00_224
Target layer: out_relu
out_relu is the final useful MobileNetV2 convolutional activation after Conv_1.
- Real prediction: Grad-CAM targets P(real)
- Fake prediction: Grad-CAM targets 1 - P(real)
- Heatmaps are generated only from gradients of the loaded trained model.
- If Grad-CAM fails, prediction still works and the UI shows a clear unavailable state with hidden technical details.
- No placeholder, random, or fabricated heatmaps are used.
Existing evaluation information
The repository’s saved test metrics report approximately:
Metric	Value
Accuracy	64.52%
AUC	78.02%
Precision	79.18%
Recall	38.70%


These should be presented carefully: the model has useful discrimination according to AUC, but recall is relatively low, meaning it may miss a meaningful number of fake images. This reinforces the responsible-AI warning: it is a research/demo classifier, not an authoritative forensic system.
Current strengths
- Real trained model is integrated.
- Prediction and Grad-CAM work from the same model/image pipeline.
- The UI is polished and presentation-ready.
- No deceptive “AI certainty” language.
- Clean separation between UI and inference service.
- Easy future path to replace local model calls with a FastAPI/cloud API.
- Good handling for unavailable model, invalid uploads, and Grad-CAM failure.
Current limitations
- The model is an image-level binary classifier, not identity verification.
- It cannot validate Aadhaar, government IDs, or confirm whether a person exists.
- Analysis history is session-only; it is not stored in a database.
- There is no user authentication or multi-user support.
- Inference runs locally, so model startup can take time.
- Grad-CAM explains model attention, not actual manipulated pixels or forensic proof.
- Current model evaluation metrics indicate room for training and dataset improvements.
How to run
.\.venv\Scripts\python.exe -m streamlit run web\app.py --server.address 127.0.0.1
Then open the displayed local URL, typically:
http://127.0.0.1:8501
Overall: TruthLens is now a functional, explainable deepfake-classification research product—not merely an upload-and-label demo.
