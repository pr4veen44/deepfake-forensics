from pathlib import Path
def test_project_files():
    root=Path(__file__).resolve().parents[1]
    assert (root/'requirements.txt').exists()
    assert (root/'src/training/train.py').exists()
    assert (root/'src/inference/predict_video.py').exists()
    assert (root/'src/inference/predict_image.py').exists()
