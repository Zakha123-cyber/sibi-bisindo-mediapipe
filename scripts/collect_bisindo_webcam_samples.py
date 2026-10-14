"""Collect explicitly labeled BISINDO webcam stills for domain adaptation."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LETTERS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def safe_id(value, name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
        raise ValueError(f'{name} must contain only letters, numbers, underscore, or hyphen (max 40 chars)')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--interval-seconds', type=float, default=0.5,
                        help='Minimum interval between saved still images in an active trial')
    parser.add_argument('--participant-id', default='p001',
                        help='Use a pseudonymous ID; do not enter a real name')
    parser.add_argument('--session-id', default=None,
                        help='Session ID; defaults to a timestamp-based ID')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/webcam_collection/bisindo')
    parser.add_argument('--save', action='store_true',
                        help='Explicitly enable saving still images when a class trial is active')
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error('--interval-seconds must be positive')
    if not args.save:
        parser.error('This collector writes webcam images locally. Pass --save to enable capture.')
    try:
        participant_id = safe_id(args.participant_id, 'participant-id')
        session_id = safe_id(args.session_id or datetime.now().strftime('session_%Y%m%d_%H%M%S'),
                             'session-id')
    except ValueError as error:
        parser.error(str(error))

    import cv2

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    participant_dir = output_dir / participant_id
    session_dir = participant_dir / session_id
    if session_dir.exists() and any(session_dir.iterdir()):
        raise SystemExit(f'Session directory already contains files: {session_dir}')
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / 'manifest.csv'

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
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fieldnames = ['path', 'language', 'label', 'participant_id', 'session_id', 'trial_id',
                  'captured_at_utc', 'width', 'height', 'inference_mirrored']
    trial_counts = {}
    frame_counts = {}
    active_label = None
    active_trial = None
    last_saved = 0.0
    frame_index = 0
    session_started = datetime.now(timezone.utc).isoformat()
    print(f'Opened {width}x{height}; saving under {session_dir}')
    print('Only saves while a letter trial is active. Form and hold a pose, then press its A-Z key.')
    print('Press SPACE to end trial and return to neutral; press Esc to finish session.')
    print('Only still JPG images are saved; frames are kept unmirrored for model training.')

    try:
        with manifest_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()
            while True:
                ok, frame_bgr = camera.read()
                if not ok:
                    print('Could not read a frame from the webcam.')
                    break
                now = time.monotonic()
                should_save = active_label is not None and (now - last_saved >= args.interval_seconds)
                if should_save:
                    frame_index += 1
                    class_dir = session_dir / active_label / active_trial
                    class_dir.mkdir(parents=True, exist_ok=True)
                    filename = f'frame_{frame_index:06d}.jpg'
                    image_path = class_dir / filename
                    if not cv2.imwrite(str(image_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError(f'Could not write webcam image: {image_path}')
                    relative_path = image_path.relative_to(output_dir).as_posix()
                    writer.writerow({
                        'path': relative_path,
                        'language': 'bisindo',
                        'label': active_label,
                        'participant_id': participant_id,
                        'session_id': session_id,
                        'trial_id': active_trial,
                        'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                        'width': width,
                        'height': height,
                        'inference_mirrored': False,
                    })
                    stream.flush()
                    frame_counts[active_label] = frame_counts.get(active_label, 0) + 1
                    last_saved = now

                preview = cv2.flip(frame_bgr, 1)
                status = f'CAPTURING {active_label} | trial {active_trial} | saved {frame_counts.get(active_label, 0)}' \
                    if active_label else 'NEUTRAL / NOT SAVING'
                lines = [status,
                         'Form pose, tekan tombol huruf A-Z untuk mulai; SPACE berhenti',
                         'Esc selesai | preview mirror; file inferensi asli (unmirrored)']
                for line_index, line in enumerate(lines):
                    y = 30 + line_index * 30
                    cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(preview, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('BISINDO labeled webcam sample collector', preview)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == 32:
                    active_label = None
                    active_trial = None
                    last_saved = 0.0
                    continue
                try:
                    letter = chr(key).upper()
                except ValueError:
                    continue
                if letter in LETTERS:
                    active_label = letter
                    trial_counts[letter] = trial_counts.get(letter, 0) + 1
                    active_trial = f'{letter}_{trial_counts[letter]:03d}_{uuid.uuid4().hex[:6]}'
                    last_saved = 0.0
    finally:
        camera.release()
        cv2.destroyAllWindows()

    summary = {
        'language': 'bisindo',
        'participant_id': participant_id,
        'session_id': session_id,
        'started_at_utc': session_started,
        'camera_device': args.device,
        'resolution': [width, height],
        'image_count_by_letter': frame_counts,
        'trial_count_by_letter': trial_counts,
        'manifest': manifest_path.name,
        'note': 'Frames are correlated samples grouped by trial_id; split by session/participant, never by individual frame.',
    }
    (session_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Collection complete. Review labels and framing before using the images for training.')


if __name__ == '__main__':
    main()
