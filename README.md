# Video Deepfake Detection Using Deep Learning

Windows / VS Code, Python 3.12, PyTorch, OpenCV, MobileNetV3, Streamlit.

## Dataset
Place videos into `data/raw/REAL/` and `data/raw/FAKE/`. Supported: mp4, avi, mov, mkv, webm. For UADFV, put original real and fake videos in the corresponding folders.

## Setup (PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Run
Run every command from the project root:
```powershell
python -m src.data.split_videos
python -m src.data.extract_faces --frames-per-video 12
python -m src.training.train --epochs 8 --batch-size 16
python -m src.training.evaluate
python -m src.inference.predict_video --video "C:\path\to\video.mp4"
streamlit run src/app.py
```

## Pipeline
1. Split source videos into train/validation/test before frame extraction (prevents frame leakage).
2. Sample frames and detect the largest face using OpenCV Haar cascade.
3. Fine-tune a pretrained MobileNetV3-Large on face crops.
4. Evaluate on held-out test images.
5. Sample frames from an uploaded video and aggregate their fake probabilities.

## Limitations
This is an educational screening prototype, not forensic proof. UADFV is small and may not generalize to unseen manipulations. The threshold is configurable. `UNCERTAIN` is returned if too few faces are detected; no detected face is not considered REAL.
