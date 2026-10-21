"""Run a reset coordinate model on webcam; no image/video recording."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import cv2
    import mediapipe as mp
    import torch
    from sibi_bisindo.coordinate_inference import CoordinatePredictor
    from sibi_bisindo.coordinate_detector import build_detector
    from webcam_sibi_classifier import draw_landmarks

    torch.set_num_threads(2)
    predictor = CoordinatePredictor(args.run_dir)
    language = predictor.selection['language']
    dataset = json.loads((args.run_dir/'dataset.json').read_text())
    extractor = build_detector(dataset, predictor.selection, ROOT, args.run_dir)
    policy_name = dataset.get('runtime_preprocessing', {}).get('version', 'baseline')
    print(f'Preprocessing: {policy_name}; unmirrored RGB; mirrored preview only.')
    if args.check_only:
        extractor.close()
        print(f"Checks passed: {language}, {predictor.selection['selected']['name']}, {len(predictor.classes)} classes")
        return
    camera = cv2.VideoCapture(args.device, cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY)
    if not camera.isOpened():
        camera.release(); extractor.close()
        raise SystemExit('Cannot open camera; try --device 1')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    history = deque(maxlen=5)
    print(f'{language}: {", ".join(predictor.classes)}; J/Z static-pose labels. Esc/q exits. Frames not saved.')
    try:
        while True:
            ok, bgr = camera.read()
            if not ok:
                break
            started = time.perf_counter()
            result = extractor.extract(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), language)
            status, stable, text = result.status, '-', 'No valid coordinates'
            inference_ms = 0.
            if result.valid:
                infer_started = time.perf_counter()
                prediction = predictor.predict(result.raw, result.height/result.width)
                inference_ms = (time.perf_counter()-infer_started)*1000
                history.append(prediction['label'])
                label, count = Counter(history).most_common(1)[0]
                if count >= 4:
                    stable = label
                text = f"Candidate {prediction['label']} | {prediction['score_type']}: {prediction['score']:.3f}"
            else:
                history.clear()
            draw_landmarks(bgr, result.raw, mp.solutions.hands.HAND_CONNECTIONS, cv2)
            preview = cv2.flip(bgr, 1)
            lines = [f'{language.upper()} | stable: {stable} | {status}', text,
                     f'Detector {result.process_ms:.1f}ms | classifier {inference_ms:.1f}ms | loop {1000*(time.perf_counter()-started):.1f}ms',
                     'Static poses; uncalibrated scores. q/Esc exits.']
            if getattr(extractor, 'last_details', None):
                details = extractor.last_details
                lines[-1] = f"Recovery {details['method']} | views {details['attempts']} | q/Esc exits"
            for index, line in enumerate(lines):
                for color, thickness in [((0, 0, 0), 4), ((255, 255, 255), 1)]:
                    cv2.putText(preview, line, (10, 25+index*25), cv2.FONT_HERSHEY_SIMPLEX, .5, color, thickness, cv2.LINE_AA)
            cv2.imshow('Coordinate reset', preview)
            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                break
    finally:
        camera.release(); extractor.close(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
