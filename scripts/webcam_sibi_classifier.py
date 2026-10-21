"""Run the selected landmark+SVM SIBI baseline on a live webcam."""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, percent):
    import numpy as np
    return float(np.percentile(values, percent)) if len(values) else None


def draw_landmarks(frame, hands, connections, cv2):
    height, width = frame.shape[:2]
    for hand in hands:
        points = [(int(round(x * width)), int(round(y * height))) for x, y, _ in hand]
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (50, 220, 70), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, (40, 80, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path,
                        default=ROOT / 'artifacts/classification-v1/sibi/selection.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/landmarks_v1.json')
    parser.add_argument('--features-dir', type=Path, default=ROOT / 'data/features/landmarks-v1-final')
    parser.add_argument('--device', type=int, default=0, help='Webcam device index')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--history', type=int, default=5,
                        help='Recent valid frames used for the preview-only stable label')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/webcam_sibi_classifier')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate model/config/dependencies without opening the webcam')
    parser.add_argument('--temporal-model', type=Path,
                        help='Optional experimental J/Z templates.npz; static baseline remains the default')
    args = parser.parse_args()
    if args.history < 1:
        parser.error('--history must be positive')

    import cv2
    import joblib
    import mediapipe as mp
    import numpy as np
    from sibi_bisindo.landmarks import HandExtractor

    selection_path = args.selection if args.selection.is_absolute() else ROOT / args.selection
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    features_dir = args.features_dir if args.features_dir.is_absolute() else ROOT / args.features_dir
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if selection.get('language') != 'sibi' or selection.get('selected_candidate') != 'svm_linear_c1':
        raise ValueError('Selection does not identify the frozen SIBI svm_linear_c1 baseline')
    if selection.get('selected_kind') != 'svm':
        raise ValueError('Selected SIBI artifact is not an SVM')
    if importlib.metadata.version('scikit-learn') != selection['scikit_learn_version']:
        raise RuntimeError('scikit-learn version differs from the selected model environment')
    if importlib.metadata.version('joblib') != selection['joblib_version']:
        raise RuntimeError('joblib version differs from the selected model environment')

    lock_path = features_dir / 'extraction_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if lock.get('config_sha256') != selection.get('feature_config_sha256'):
        raise ValueError('Feature configuration differs from the frozen SIBI model selection')
    implementation_sha = sha256_file(ROOT / 'src/sibi_bisindo/landmarks.py')
    if implementation_sha != selection.get('feature_implementation_sha256'):
        raise ValueError('Landmark feature implementation differs from the selected model')

    artifact_rel = Path(selection['artifact'].replace('\\', '/'))
    artifact_path = artifact_rel if artifact_rel.is_absolute() else ROOT / artifact_rel
    artifact_sha = sha256_file(artifact_path)
    if artifact_sha != selection.get('artifact_sha256'):
        raise ValueError('Selected classifier artifact hash differs from selection.json')
    bundle = joblib.load(artifact_path)
    classes = list(selection['classes'])
    if (bundle.get('language') != 'sibi' or bundle.get('kind') != 'svm'
            or list(bundle.get('classes', [])) != classes):
        raise ValueError('Selected classifier bundle metadata does not match SIBI selection')
    if bundle['model'].named_steps.get('svc') is None:
        raise ValueError('Expected a StandardScaler + SVC pipeline')

    temporal = None
    if args.temporal_model:
        from sibi_bisindo.temporal import MotionRecognizer
        temporal = MotionRecognizer(args.temporal_model)
    extractor = HandExtractor(config)
    if args.check_only:
        extractor.close()
        print(f"Model/config checks passed: {selection['selected_candidate']} "
              f"({len(classes)} classes), artifact sha256={artifact_sha}")
        return
    camera_backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.device, camera_backend)
    if not camera.isOpened() and platform.system() == 'Windows':
        camera.release()
        camera = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    if not camera.isOpened():
        camera.release()
        extractor.close()
        raise SystemExit(f'Could not open webcam device {args.device}. Try --device 1.')

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Opened camera {args.device}: {actual_width}x{actual_height}')
    print(f"SIBI static classes: {', '.join(classes)}")
    print('Experimental J/Z enabled; rejection thresholds are heuristic.' if temporal else 'J/Z disabled.')
    print('Inference uses unmirrored RGB frames and the frozen landmarks-v1 pipeline.')
    print('Preview is mirrored; SVM margins are not probabilities. Frames are not saved. Press q or Esc to stop.')

    connections = list(mp.solutions.hands.HAND_CONNECTIONS)
    history = deque(maxlen=args.history)
    statuses = Counter()
    predictions = Counter()
    detector_ms, classifier_ms, frame_intervals = [], [], []
    frames = 0
    started_at = datetime.now(timezone.utc).isoformat()
    session_started = time.perf_counter()
    last_frame = session_started
    fps_ema = 0.0
    motion_event = None
    motion_event_at = -float('inf')
    try:
        while True:
            ok, frame_bgr = camera.read()
            if not ok:
                print('Could not read a frame from the webcam.')
                break
            captured_at = time.perf_counter()
            frame_intervals.append(captured_at - last_frame)
            last_frame = captured_at
            frames += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            extraction = extractor.extract(frame_rgb, 'sibi', input_mirrored=False)
            statuses[extraction.status] += 1
            detector_ms.append(extraction.process_ms)
            draw_landmarks(frame_bgr, extraction.raw, connections, cv2)
            if temporal is not None:
                event = temporal.update(captured_at, extraction.raw if extraction.valid else [],
                                        extraction.height / extraction.width)
                if event:
                    motion_event, motion_event_at = event, captured_at
                    history.clear()

            predicted = None
            margin = None
            classifier_time = None
            if extraction.valid and extraction.features is not None:
                classifier_started = time.perf_counter()
                feature = extraction.features.reshape(1, -1)
                predicted = str(bundle['model'].predict(feature)[0])
                decision = np.asarray(bundle['model'].decision_function(feature)).reshape(-1)
                ordered = np.sort(decision)
                margin = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])
                classifier_time = (time.perf_counter() - classifier_started) * 1000
                classifier_ms.append(classifier_time)
                predictions[predicted] += 1
                history.append(predicted)
            else:
                history.clear()

            stable_label = None
            stable_votes = 0
            if history:
                label_counts = Counter(history)
                stable_label, stable_votes = label_counts.most_common(1)[0]
                if stable_votes < max(1, (args.history * 4 + 4) // 5):
                    stable_label = None

            now = time.perf_counter()
            frame_fps = 1.0 / max(now - captured_at, 1e-9)
            fps_ema = frame_fps if fps_ema == 0 else 0.85 * fps_ema + 0.15 * frame_fps
            if predicted:
                candidate_text = f'prediksi: {predicted} | SVM margin {margin:.3f} (bukan probabilitas)'
                stable_text = (f'stabil: {stable_label} ({stable_votes}/{len(history)})'
                               if stable_label else f'stabil: menunggu ({stable_votes}/{len(history)})')
            elif extraction.status == 'unexpected_extra_hand':
                candidate_text = 'dua tangan terdeteksi; baseline SIBI ini hanya menerima satu'
                stable_text = 'stabil: —'
            elif extraction.status == 'no_hand':
                candidate_text = 'tangan tidak terdeteksi'
                stable_text = 'stabil: —'
            else:
                candidate_text = f'fitur tidak valid: {extraction.status}'
                stable_text = 'stabil: —'

            lines = [
                f'SIBI | {extraction.status} | tangan: {len(extraction.raw)}',
                candidate_text,
                stable_text,
                f'MediaPipe: {extraction.process_ms:.1f} ms | FPS: {fps_ema:.1f} | {actual_width}x{actual_height}',
                'q/Esc keluar; preview mirror, inferensi unmirrored',
            ]
            if temporal is not None:
                active = motion_event if captured_at-motion_event_at <= 1. else None
                lines.insert(1, f"HASIL: {active['label']} | gerakan eksperimental" if active else
                             f"HASIL: {stable_label or '-'} | menunggu gerakan J/Z")
            preview = cv2.flip(frame_bgr, 1)
            for line_index, line in enumerate(lines):
                y = 28 + line_index * 27
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.56, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.56, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('SIBI webcam classifier', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        camera.release()
        extractor.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - session_started
    summary = {
        'started_at_utc': started_at,
        'language': 'sibi',
        'selection_candidate': selection['selected_candidate'],
        'classifier_artifact_sha256': artifact_sha,
        'feature_config_sha256': selection['feature_config_sha256'],
        'feature_implementation_sha256': implementation_sha,
        'camera_device': args.device,
        'requested_resolution': [args.width, args.height],
        'actual_resolution': [actual_width, actual_height],
        'preview_mirrored': True,
        'inference_mirrored': False,
        'frames': frames,
        'duration_seconds': round(elapsed, 3),
        'throughput_fps': round(frames / elapsed, 2) if elapsed else None,
        'status_frames': dict(statuses),
        'prediction_frames': dict(predictions),
        'detector_ms': {
            'median': round(statistics.median(detector_ms), 2) if detector_ms else None,
            'p95': round(percentile(detector_ms, 95), 2) if detector_ms else None,
        },
        'classifier_ms': {
            'median': round(statistics.median(classifier_ms), 3) if classifier_ms else None,
            'p95': round(percentile(classifier_ms, 95), 3) if classifier_ms else None,
        },
        'capture_fps_median': round(1 / statistics.median(frame_intervals), 2) if frame_intervals else None,
        'classes': classes,
        'temporal_model': str(args.temporal_model) if args.temporal_model else None,
        'note': 'Frame-level webcam diagnostics only, not independent pose trials. SVM margins are uncalibrated decision scores. Frames were not saved.',
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'webcam_sibi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Session summary saved to {output_path}')


if __name__ == '__main__':
    main()
