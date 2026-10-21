"""Webcam preview and labeled pose trials for the new 63D landmark candidate.

No images or video are saved. Press a supported letter to label a short trial
after forming its pose; press Esc to exit (Q is a supported letter).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def trial_prediction(stable_counts):
    ranked = stable_counts.most_common()
    winner = ranked[0][0] if ranked else '__reject__'
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        winner = '__reject__'
    return None if winner == '__reject__' else winner


def finish_trial(trial, now, completed):
    stable_counts = trial['stable_counts']
    predicted = trial_prediction(stable_counts)
    report = {
        'expected': trial['expected'], 'completed': completed,
        'duration_seconds': round(now-trial['started'], 3),
        'frames': trial['frames'], 'frame_predictions': dict(trial['raw_counts']),
        'stable_predictions': dict(stable_counts), 'statuses': dict(trial['statuses']),
        'trial_prediction': predicted, 'correct': predicted == trial['expected'],
    }
    if 'comparison' in trial:
        other = trial['comparison']
        other_predicted = trial_prediction(other['stable_counts'])
        report['comparison'] = {
            'orientation': other['orientation'],
            'frame_predictions': dict(other['raw_counts']),
            'stable_predictions': dict(other['stable_counts']),
            'trial_prediction': other_predicted,
            'correct': other_predicted == trial['expected'],
        }
    return report


def draw_landmarks(frame, hands, connections, cv2):
    height, width = frame.shape[:2]
    for hand in hands:
        points = [(int(round(x*width)), int(round(y*height))) for x, y, _ in hand]
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (50, 220, 70), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, (40, 80, 255), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=ROOT / 'artifacts/asl-source-sibi-candidate-v1_20261020_093347')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/webcam_npy_landmarks_v1.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--history', type=int, default=5)
    parser.add_argument('--landmark-orientation', choices=['original', 'mirror-x'], default='original',
                        help='Explicit x orientation before shared feature normalization')
    parser.add_argument('--compare-mirror', action='store_true',
                        help='Compare both landmark x orientations on the same frame and trial')
    parser.add_argument('--trial-seconds', type=float, default=3)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/webcam_npy_landmarks')
    parser.add_argument('--check-only', action='store_true',
                        help='Verify model, preprocessing and detector without opening a camera')
    args = parser.parse_args()
    if min(args.width, args.height, args.history) < 1 or not 0 < args.trial_seconds <= 60:
        parser.error('Dimensions/history must be positive; trial-seconds must be in (0, 60]')

    import cv2
    import mediapipe as mp
    import numpy as np
    from sibi_bisindo.npy_inference import (PredictionWindow, RawHandDetector,
                                           load_selected_model, predict_hands)
    from sibi_bisindo.npy_landmarks import sha256_file

    run_dir = args.run_dir.resolve()
    bundle, selection = load_selected_model(run_dir)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    detector = RawHandDetector(config)
    try:
        # Warm the detector with a synthetic blank image, without camera access.
        detector.detect(np.zeros((args.height, args.width, 3), dtype=np.uint8))
        if args.check_only:
            print(f"Checks passed: {selection['selected_candidate']}, {len(selection['classes'])} classes, 63 features.")
            print('Model SHA and shared preprocessing verified; camera was not opened.')
            return
        run_preview(args, run_dir, bundle, selection, detector, cv2, mp, np,
                    PredictionWindow, predict_hands, sha256_file)
    finally:
        detector.close()


def run_preview(args, run_dir, bundle, selection, detector, cv2, mp, np,
                window_class, predict_hands, sha256_file):
    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.device, backend)
    if not camera.isOpened() and platform.system() == 'Windows':
        camera.release()
        camera = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(f'Could not open webcam {args.device}. Try --device 1.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    resolution = [int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    window = window_class(args.history)
    comparison_window = window_class(args.history)
    other_orientation = 'mirror-x' if args.landmark_orientation == 'original' else 'original'
    classes = selection['classes']
    connections = list(mp.solutions.hands.HAND_CONNECTIONS)
    print(f"Opened {resolution[0]}x{resolution[1]}; selected {selection['selected_candidate']}.")
    print(f"Supported labels: {', '.join(classes)}; ASL-source labels, SIBI pose mapping unverified.")
    print('RGB inference is unmirrored; preview is mirrored. Images/video are not saved.')
    print(f'Primary landmark orientation: {args.landmark_orientation}; mirror comparison: {args.compare_mirror}')
    print('Form a pose, press its letter for a labeled trial, and hold until the trial ends.')
    print('Space: cancel/reset; Esc: exit. Q selects the letter Q, not quit.')
    statuses, predictions = Counter(), Counter()
    detector_times, classifier_times, pipeline_times = [], [], []
    trial, trials = None, []
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    previous_capture, fps_ema, frames = None, 0.0, 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print('Could not read a camera frame.')
                break
            captured = time.perf_counter()
            frames += 1
            if previous_capture is not None:
                gap = captured-previous_capture
                if gap > 1:
                    window.reset()
                    comparison_window.reset()
                fps = 1/max(gap, 1e-9)
                fps_ema = fps if fps_ema == 0 else fps_ema*0.85 + fps*0.15
            previous_capture = captured
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detect_started = time.perf_counter()
            hands = detector.detect(rgb)
            detect_ms = (time.perf_counter()-detect_started)*1000
            classify_started = time.perf_counter()
            result = predict_hands(bundle, hands, orientation=args.landmark_orientation)
            other_result = (predict_hands(bundle, hands, orientation=other_orientation)
                            if args.compare_mirror else None)
            classify_ms = (time.perf_counter()-classify_started)*1000
            stable = window.update(result['letter'])
            other_stable = comparison_window.update(other_result['letter']) if other_result else None
            now = time.perf_counter()
            process_ms = (now-captured)*1000
            detector_times.append(detect_ms)
            pipeline_times.append(process_ms)
            if result['letter'] is not None:
                classifier_times.append(classify_ms)
                predictions[result['letter']] += 1
            statuses[result['status']] += 1

            if trial is not None:
                trial['frames'] += 1
                trial['raw_counts'][result['letter'] or '__reject__'] += 1
                trial['stable_counts'][stable or '__reject__'] += 1
                trial['statuses'][result['status']] += 1
                if other_result:
                    trial['comparison']['raw_counts'][other_result['letter'] or '__reject__'] += 1
                    trial['comparison']['stable_counts'][other_stable or '__reject__'] += 1
                if now-trial['started'] >= args.trial_seconds:
                    finished = finish_trial(trial, now, completed=True)
                    trials.append(finished)
                    print(f"Trial {finished['expected']} -> {finished['trial_prediction'] or 'rejected'} "
                          f"({'correct' if finished['correct'] else 'incorrect'}), frames={finished['frames']}", flush=True)
                    if 'comparison' in finished:
                        compared = finished['comparison']
                        print(f"  comparison {compared['orientation']} -> {compared['trial_prediction'] or 'rejected'} "
                              f"({'correct' if compared['correct'] else 'incorrect'})", flush=True)
                    trial = None

            draw_landmarks(frame, hands, connections, cv2)
            preview = cv2.flip(frame, 1)
            status_text = {'ok': '1 tangan', 'no_hand': 'tangan tidak terdeteksi',
                           'unexpected_extra_hand': 'gunakan 1 tangan',
                           'invalid_geometry': 'landmark tidak valid'}[result['status']]
            score_text = f"skor {result['score']:.3f} (belum dikalibrasi)" if result['score'] is not None else '-'
            trial_text = (f"Uji {trial['expected']}: tahan {max(0, args.trial_seconds-(now-trial['started'])):.1f}s lagi"
                          if trial else 'Bentuk pose lalu tekan hurufnya untuk uji berlabel')
            lines = [
                f"Kandidat 63 fitur | {status_text} | {args.landmark_orientation}",
                f"Prediksi: {result['letter'] or '-'} | Stabil: {stable or '-'} | {score_text}",
                f'Proses: {process_ms:.1f} ms | FPS: {fps_ema:.1f}',
                trial_text,
                'Label A-I, K-Y | Space reset | Esc keluar (Q = huruf Q)',
            ]
            if other_result:
                lines.insert(2, f"Pembanding {other_orientation}: {other_result['letter'] or '-'} | Stabil: {other_stable or '-'}")
            for index, line in enumerate(lines):
                y = 26 + index*26
                cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Uji webcam kandidat 63 fitur', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                if trial is not None:
                    trials.append(finish_trial(trial, time.perf_counter(), completed=False))
                    trial = None
                window.reset()
                comparison_window.reset()
            elif chr(key).upper() in classes and trial is None:
                window.reset()
                comparison_window.reset()
                trial = {'expected': chr(key).upper(), 'started': time.perf_counter(), 'frames': 0,
                         'raw_counts': Counter(), 'stable_counts': Counter(), 'statuses': Counter()}
                if args.compare_mirror:
                    trial['comparison'] = {'orientation': other_orientation,
                                           'raw_counts': Counter(), 'stable_counts': Counter()}
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if trial is not None:
        trials.append(finish_trial(trial, time.perf_counter(), completed=False))
    duration = time.perf_counter()-started

    def latency_summary(values):
        return {'median': round(float(np.median(values)), 3) if values else None,
                'p95': round(float(np.percentile(values, 95)), 3) if values else None}

    completed = [trial for trial in trials if trial['completed']]
    summary = {
        'started_at_utc': started_at, 'run_dir': str(run_dir),
        'candidate': selection['selected_candidate'], 'artifact_sha256': selection['artifact_sha256'],
        'source_language': 'asl', 'target_language': 'sibi', 'pose_mapping': 'unverified',
        'feature_version': selection['feature_version'], 'classes': classes,
        'detector_config': json.loads(args.config.read_text(encoding='utf-8')),
        'detector_config_sha256': sha256_file(args.config),
        'camera_device': args.device, 'resolution': resolution,
        'preview_mirrored': True, 'inference_mirrored': False, 'frames': frames,
        'landmark_orientation': args.landmark_orientation,
        'mirror_comparison_enabled': args.compare_mirror,
        'comparison_orientation': other_orientation if args.compare_mirror else None,
        'duration_seconds': round(duration, 3), 'throughput_fps': round(frames/duration, 2),
        'status_frames': dict(statuses), 'prediction_frames': dict(predictions),
        'detector_ms': latency_summary(detector_times), 'classifier_ms': latency_summary(classifier_times),
        'pipeline_ms': latency_summary(pipeline_times),
        'stabilization': {'history_frames': args.history, 'minimum_votes': window.minimum_votes},
        'trial_seconds': args.trial_seconds, 'trials': trials,
        'completed_trials': len(completed), 'correct_completed_trials': sum(t['correct'] for t in completed),
        'correct_comparison_trials': (sum(t['comparison']['correct'] for t in completed)
                                      if args.compare_mirror else None),
        'note': 'Local exploratory trials, not a final independent signer/session evaluation. No images/video saved. Scores are uncalibrated.',
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output_dir / f'webcam_npy_{stamp}.json'
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f"Completed trials: {len(completed)}; correct: {summary['correct_completed_trials']}")
    if args.compare_mirror:
        print(f"Comparison {other_orientation}: correct={summary['correct_comparison_trials']}/{len(completed)}")
    print(f'Summary saved to {output}')


if __name__ == '__main__':
    main()
