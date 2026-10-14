"""Inspect MediaPipe hand detection on a live webcam without saving frames.

Example:
    python scripts/webcam_detector_diagnostic.py --config configs/landmarks_bisindo_kaggle_v2_conf03.json

Press q or Esc to close the preview. A JSON summary is written locally; images
and video are never recorded.
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

import cv2
import numpy as np

from sibi_bisindo.landmarks import HandExtractor


def draw_landmarks(frame, raw, connections):
    height, width = frame.shape[:2]
    for hand in raw:
        points = []
        for x, y, _ in hand:
            points.append((int(round(x * width)), int(round(y * height))))
        for start, end in connections:
            if 0 <= start < len(points) and 0 <= end < len(points):
                cv2.line(frame, points[start], points[end], (50, 220, 70), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, (40, 80, 255), -1, cv2.LINE_AA)


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs/landmarks_bisindo_kaggle_v2_conf03.json')
    parser.add_argument('--device', type=int, default=0, help='Webcam device index (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Requested capture width')
    parser.add_argument('--height', type=int, default=480, help='Requested capture height')
    parser.add_argument('--segment-seconds', type=float, default=10,
                        help='Group session metrics into time segments (default: 10)')
    parser.add_argument('--language', choices=['bisindo', 'sibi'], default='bisindo')
    parser.add_argument('--no-mirror-preview', action='store_true',
                        help='Show the preview without mirroring; inference is always unmirrored')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/webcam_diagnostics')
    args = parser.parse_args()
    if args.segment_seconds <= 0:
        parser.error('--segment-seconds must be greater than zero')

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding='utf-8'))
    extractor = HandExtractor(config)
    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.device, backend)
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
    print(f'Opened device {args.device}: {actual_width}x{actual_height}')
    print('Inference uses unmirrored RGB frames. Preview mirroring does not change inference.')
    print('Press q or Esc to stop. Frames are not saved.')

    try:
        import mediapipe as mp
        connections = list(mp.solutions.hands.HAND_CONNECTIONS)
        hand_counts = Counter()
        statuses = Counter()
        process_times = []
        capture_times = []
        segments = {}
        frame_count = 0
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        previous = started
        display_fps = 0.0
        preview_mirror = not args.no_mirror_preview

        while True:
            read_started = time.perf_counter()
            ok, frame_bgr = camera.read()
            if not ok:
                print('Could not read a frame from the webcam.')
                break
            captured_at = time.perf_counter()
            capture_times.append(captured_at)
            frame_count += 1
            segment_index = max(0, int((captured_at - started) // args.segment_seconds))
            segment_key = f'segment_{segment_index + 1}'
            segment = segments.setdefault(segment_key, {
                'start_seconds': round(segment_index * args.segment_seconds, 2),
                'end_seconds': round((segment_index + 1) * args.segment_seconds, 2),
                'frames': 0,
                'hand_count_frames': Counter(),
                'status_frames': Counter(),
                'detector_ms': [],
                'camera_read_ms': [],
                'capture_timestamps': [],
            })
            read_ms = (captured_at - read_started) * 1000
            segment['frames'] += 1
            segment['camera_read_ms'].append(read_ms)
            segment['capture_timestamps'].append(captured_at)

            # OpenCV supplies BGR; MediaPipe's Hands API expects RGB.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = extractor.extract(frame_rgb, args.language, input_mirrored=False)
            process_times.append(result.process_ms)
            hand_counts[str(len(result.raw))] += 1
            statuses[result.status] += 1
            segment['hand_count_frames'][str(len(result.raw))] += 1
            segment['status_frames'][result.status] += 1
            segment['detector_ms'].append(result.process_ms)
            draw_landmarks(frame_bgr, result.raw, connections)

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous, 1e-9)
            display_fps = instantaneous_fps if display_fps == 0 else 0.85 * display_fps + 0.15 * instantaneous_fps
            previous = now
            lines = [
                f"{result.status} | hands: {len(result.raw)} | detector: {result.process_ms:.1f} ms",
                f'frame: {result.width}x{result.height} | FPS: {display_fps:.1f} | mode: {args.language}',
                f'{segment_key} ({segment["start_seconds"]:.0f}-{segment["end_seconds"]:.0f}s) | q/Esc to quit',
            ]
            for index, line in enumerate(lines):
                y = 28 + index * 28
                cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame_bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 1, cv2.LINE_AA)

            preview = cv2.flip(frame_bgr, 1) if preview_mirror else frame_bgr
            cv2.imshow('MediaPipe webcam detector diagnostic', preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

        elapsed = time.perf_counter() - started
        intervals = np.diff(capture_times) if len(capture_times) > 1 else np.array([])
        segment_summaries = {}
        for key, segment in segments.items():
            times = segment.pop('capture_timestamps')
            segment_intervals = np.diff(times) if len(times) > 1 else np.array([])
            segment_summaries[key] = {
                'start_seconds': segment['start_seconds'],
                'end_seconds': segment['end_seconds'],
                'frames': segment['frames'],
                'hand_count_frames': dict(segment['hand_count_frames']),
                'status_frames': dict(segment['status_frames']),
                'camera_read_ms': {
                    'median': round(percentile(segment['camera_read_ms'], 50), 2),
                    'p95': round(percentile(segment['camera_read_ms'], 95), 2),
                },
                'frame_interval_ms': {
                    'median': round(percentile(segment_intervals * 1000, 50), 2) if len(segment_intervals) else None,
                    'p95': round(percentile(segment_intervals * 1000, 95), 2) if len(segment_intervals) else None,
                },
                'detector_ms': {
                    'median': round(percentile(segment['detector_ms'], 50), 2),
                    'p95': round(percentile(segment['detector_ms'], 95), 2),
                },
            }
        summary = {
            'started_at_utc': started_at_utc,
            'config': str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
            'config_version': config.get('version'),
            'language': args.language,
            'camera_device': args.device,
            'requested_resolution': [args.width, args.height],
            'actual_resolution': [actual_width, actual_height],
            'preview_mirrored': preview_mirror,
            'inference_mirrored': False,
            'frames': frame_count,
            'duration_seconds': round(elapsed, 3),
            'capture_fps_median': round(1.0 / float(np.median(intervals)), 2) if len(intervals) else None,
            'detector_ms': {
                'median': round(percentile(process_times, 50), 2) if process_times else None,
                'p95': round(percentile(process_times, 95), 2) if process_times else None,
            },
            'segments': segment_summaries,
            'hand_count_frames': dict(hand_counts),
            'status_frames': dict(statuses),
            'note': 'Frames and images were not saved; counts are frame-level, not independent pose trials.',
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f'webcam_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f'Summary saved to {output}')
    finally:
        camera.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
