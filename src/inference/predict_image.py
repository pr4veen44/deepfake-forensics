import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.models import build_model
from src.utils import project_root


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f'Image not found: {image_path}')

    root = project_root()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
    if not checkpoint_path.exists():
        raise SystemExit('Checkpoint missing. Train the model first.')

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit('OpenCV could not read this image.')

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces):
        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        px, py = int(.18 * width), int(.18 * height)
        image = image[max(0, y - py):min(image.shape[0], y + height + py),
                      max(0, x - px):min(image.shape[1], x + width + px)]
        input_source = 'largest_detected_face'
    else:
        input_source = 'full_image_no_face_detected'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    fake_index = checkpoint.get('classes', ['FAKE', 'REAL']).index('FAKE')
    threshold = args.threshold if args.threshold is not None else checkpoint.get('threshold', .5)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = TF(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        fake_probability = float(torch.softmax(model(tensor), 1)[0, fake_index].item())

    result = {
        'image': str(image_path.resolve()),
        'input_source': input_source,
        'threshold': float(threshold),
        'fake_probability': round(fake_probability, 4),
        'image_prediction': 'LIKELY_FAKE' if fake_probability >= threshold else 'LIKELY_REAL',
        'reason': 'Probability from the selected image region; this is not forensic proof.',
    }
    output_dir = root / 'outputs/reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / f'{image_path.stem}_image_prediction.json'
    report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print('Report saved:', report)


if __name__ == '__main__':
    main()