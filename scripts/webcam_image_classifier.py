"""Preview BISINDO image-classifier predictions from a live webcam."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
LETTERS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/bisindo-image-run-01/best.pt')
    parser.add_argument('--device', type=int, default=0, help='Webcam device index')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--history', type=int, default=5,
                        help='Number of recent frames used to show a stable prediction')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/webcam_image_classifier')
    args = parser.parse_args()
    if args.history < 1:
        parser.error('--history must be at least 1')

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torchvision import models, transforms

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not checkpoint_path.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'mobilenet_v3_small' or checkpoint.get('class_names') != LETTERS:
        raise ValueError('Checkpoint is not a supported 26-class MobileNetV3-Small BISINDO model')
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LETTERS))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    use_cuda = torch.cuda.is_available()
    inference_device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(inference_device).eval()
    transform = transforms.Compose([
        transforms.Resize((checkpoint['image_size'], checkpoint['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=checkpoint['normalization']['mean'],
                             std=checkpoint['normalization']['std']),
    ])

    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.device, backend)
    if not camera.isOpened() and platform.system() == 'Windows':
        camera.release()
        camera = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f'Could not open webcam device {args.device}. Try --device 1.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Opened webcam {args.device}: {actual_width}x{actual_height}; inference={inference_device}')
    print('Inference uses unmirrored frames; preview is mirrored. Press q or Esc to stop.')
    print('Scores are softmax outputs, not calibrated probabilities. Frames are not saved.')

    probabilities = deque(maxlen=args.history)
    inference_ms = []
    frame_count = 0
    started_at = datetime.now(timezone.utc).isoformat()
    session_started = time.perf_counter()
    previous_frame_end = session_started
    fps_ema = 0.0
    top_counts = Counter()
    try:
        while True:
            ok, frame_bgr = camera.read()
            if not ok:
                print('Could not read a frame from the webcam.')
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            tensor = transform(image).unsqueeze(0).to(inference_device)
            infer_started = time.perf_counter()
            with torch.inference_mode():
                scores = torch.softmax(model(tensor)[0], dim=0)
            elapsed_ms = (time.perf_counter() - infer_started) * 1000
            inference_ms.append(elapsed_ms)
            frame_count += 1
            values, indices = torch.topk(scores, k=3)
            top_label = LETTERS[int(indices[0])]
            top_score = float(values[0])
            top_counts[top_label] += 1
            probabilities.append(scores.detach().cpu().numpy())
            mean_scores = np.mean(np.stack(probabilities), axis=0)
            stable_index = int(np.argmax(mean_scores))
            stable_label = LETTERS[stable_index]
            stable_votes = sum(int(np.argmax(frame_scores)) == stable_index for frame_scores in probabilities)
            stable_text = stable_label if stable_votes >= max(1, (args.history * 4 + 4) // 5) else 'menunggu stabil'

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_frame_end, 1e-9)
            fps_ema = instantaneous_fps if fps_ema == 0 else 0.85 * fps_ema + 0.15 * instantaneous_fps
            previous_frame_end = now
            top3 = '  '.join(f'{LETTERS[int(i)]}:{float(v):.2f}' for v, i in zip(values, indices))
            lines = [
                f'BISINDO | frame: {top_label} (score {top_score:.2f})',
                f'Prediksi stabil ({len(probabilities)} frame): {stable_text} | suara {stable_votes}/{len(probabilities)}',
                f'Top-3: {top3}',
                f'inferensi: {elapsed_ms:.1f} ms | FPS: {fps_ema:.1f} | {actual_width}x{actual_height} | q/Esc',
            ]
            # Mirror the camera preview first, then draw readable overlays.
            # The original unmirrored frame above remains the inference input.
            preview = cv2.flip(frame_bgr, 1)
            for line_index, line in enumerate(lines):
                y = 30 + line_index * 30
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('BISINDO image-classifier webcam preview', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    duration = time.perf_counter() - session_started
    intervals_fps = frame_count / duration if duration > 0 else 0.0
    summary = {
        'started_at_utc': started_at,
        'checkpoint': str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        'checkpoint_sha256': sha256_file(checkpoint_path),
        'checkpoint_epoch': checkpoint['epoch'],
        'validation_macro_f1': checkpoint['validation_macro_f1'],
        'camera_device': args.device,
        'resolution': [actual_width, actual_height],
        'inference_device': str(inference_device),
        'preview_mirrored': True,
        'inference_mirrored': False,
        'frames': frame_count,
        'duration_seconds': round(duration, 3),
        'throughput_fps': round(intervals_fps, 2),
        'inference_ms': {
            'median': round(statistics.median(inference_ms), 2) if inference_ms else None,
            'p95': round(percentile(inference_ms, 95), 2) if inference_ms else None,
        },
        'top_prediction_frame_counts': dict(top_counts),
        'smoothing_history_frames': args.history,
        'note': 'Exploratory frame-level webcam preview; not independent pose trials. Softmax scores are not calibrated probabilities. No frames were saved.',
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'webcam_image_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Session summary saved to {output_path}')


if __name__ == '__main__':
    main()
