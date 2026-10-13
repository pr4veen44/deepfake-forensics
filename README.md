# Video Deepfake Detection Using Deep Learning

A deep learning-based video analysis system that detects potentially manipulated facial content using MobileNetV3-Large. Extracts faces from video frames, classifies them as REAL or FAKE, and aggregates predictions to generate a video-level assessment.

---

## Overview

This project implements an end-to-end video classification pipeline using computer vision and transfer learning. Videos are sampled into frames, faces are detected and cropped, and a pretrained MobileNetV3-Large model analyzes each face for potential manipulation.

Frame-level fake probabilities are aggregated to produce an overall video prediction. The system also generates evaluation metrics and frame-level reports through a Streamlit interface.

---

## Tech Stack

- **Deep Learning** — PyTorch, Torchvision
- **Model Architecture** — MobileNetV3-Large (ImageNet pretrained)
- **Computer Vision** — OpenCV, Haar Cascade
- **Data Processing** — NumPy, Pandas, Pillow
- **Evaluation** — scikit-learn, Matplotlib
- **Frontend** — Streamlit
- **Dataset** — UADFV

---

## Project Structure

```text
deepfake-video-detection/
├── data/
│   ├── raw/
│   │   ├── REAL/
│   │   └── FAKE/
│   ├── splits/
│   └── processed/
├── checkpoints/
│   └── best_model.pt
├── outputs/
│   ├── metrics/
│   └── reports/
├── src/
│   ├── data/
│   │   ├── split_videos.py
│   │   └── extract_faces.py
│   ├── models.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict_video.py
│   ├── app.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## Running the Project

### Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset Preparation

Place real and manipulated videos in `data/raw/REAL/` and `data/raw/FAKE/`.

```powershell
python -m src.data.split_videos
python -m src.data.extract_faces --frames-per-video 12
```

### Train and Evaluate

```powershell
python -m src.training.train --epochs 8 --batch-size 16
python -m src.training.evaluate
```

### Run Video Prediction

```powershell
python -m src.inference.predict_video --video "path\to\video.mp4"
```

### Launch the Interface

```powershell
streamlit run src/app.py
```

---

## How It Works

- Videos are split into training, validation, and test sets before frame extraction to prevent data leakage.
- OpenCV samples frames and detects the largest visible face.
- Face crops are resized to 224 × 224 and normalized.
- MobileNetV3-Large classifies each face crop as REAL or FAKE.
- Frame-level fake probabilities are aggregated into a video-level prediction.
- Evaluation metrics and prediction reports are saved for analysis.

---

## Model Details

| Component | Configuration |
|---|---|
| Architecture | MobileNetV3-Large |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 RGB |
| Classification | Binary: REAL / FAKE |
| Loss function | Cross-Entropy Loss |
| Optimizer | AdamW |
| Initial training | Frozen backbone, trainable classifier |
| Video aggregation | Mean fake probability |

---

## Features

- Automated video frame sampling and face extraction
- Transfer learning with a pretrained CNN
- Frame-level and video-level predictions
- Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- JSON prediction reports with frame timestamps
- Interactive Streamlit interface
- CPU-compatible training and inference

---

## Evaluation

The model is evaluated on a held-out test split using standard classification metrics. Videos are split before extracting frames so frames from the same source video cannot leak across the dataset splits.

---

## Limitations

- UADFV is relatively small and may not represent newer manipulation methods.
- Haar Cascade may miss faces under challenging conditions.
- Mean probability aggregation does not explicitly model temporal inconsistencies.
- Predictions are estimates, not forensic proof; performance on unseen datasets may differ.

