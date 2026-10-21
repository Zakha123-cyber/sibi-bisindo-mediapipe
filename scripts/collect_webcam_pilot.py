"""Collect participant-separated SIBI/BISINDO pilot trials without model feedback."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sibi_bisindo.pilot import (TrialWriter, completed_counts, lock_collection,
                                read_pose_review, sha, utc_now, validate_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/webcam_pilot_v1.json')
    parser.add_argument('--mode', choices=['sibi', 'bisindo'], required=True)
    parser.add_argument('--participant-id', required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--lighting', default='unspecified', help='e.g. daylight or room_lamp')
    parser.add_argument('--background', default='unspecified', help='e.g. plain_wall or room')
    parser.add_argument('--hand', choices=['right', 'left', 'both', 'unspecified'], default='unspecified')
    parser.add_argument('--save', action='store_true', help='Enable local JPEG collection')
    parser.add_argument('--consent', action='store_true', help='Participant agreed to local image collection')
    parser.add_argument('--check-only', action='store_true', help='Validate protocol without camera or writes')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    if args.participant_id not in config['participants'] or args.session_id not in config['sessions']:
        parser.error('Use a participant/session ID from configs/webcam_pilot_v1.json')
    if min(args.width, args.height) < 1:
        parser.error('Resolution must be positive')
    output = (ROOT / config['output_dir']).resolve()
    reviews = {}
    for label in config['languages'][args.mode]['classes']:
        reviews[label] = read_pose_review(ROOT / config['pose_review_file'], args.mode, label)
    role = config['participants'][args.participant_id]
    print(f'{args.mode}: {config["languages"][args.mode]["classes"]}; {args.participant_id}/{args.session_id} -> {role}')
    print('Pose references are candidates. Pending pose/trial labels require review before training.')
    if args.check_only:
        print(f'Protocol check passed. No camera opened, no files written. Destination: {output}')
        return
    if not args.save or not args.consent:
        parser.error('Collection saves local images; pass --save --consent after participant agreement')

    import cv2
    backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.device, backend)
    if not camera.isOpened() and platform.system() == 'Windows':
        camera.release()
        camera = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    if not camera.isOpened():
        camera.release()
        raise SystemExit('Could not open camera. Try --device 1.')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    resolution = [int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    context = {'consent_recorded': True, 'camera_device': args.device, 'resolution': resolution,
               'lighting': args.lighting, 'background': args.background, 'signing_hand': args.hand,
               'preview_mirrored': config['preview_mirrored'],
               'collector_sha256': sha(Path(__file__).read_bytes())}
    session_dir = output / args.mode / args.participant_id / args.session_id
    writer, pending, deadline, last_saved, record_started = None, None, 0.0, 0.0, 0.0
    protocol_locked = False
    session_started = utc_now()
    counts = completed_counts(session_dir)
    print(f"Letter key: prepare {config['countdown_seconds']:g} seconds, then hold {config['capture_seconds']:g} seconds. Digit 1-4: negative trial.")
    print('1=no_hand; 2=neutral; 3=non_sign_pose; 4=transition. Space cancels. Esc exits.')
    print('Frames are saved unmirrored, even when no hands are detected. No classifier or detector filters collection.')
    try:
        lock_collection(output, config)
        protocol_locked = True
        session_dir.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = camera.read()
            if not ok:
                print('Camera read failed.')
                break
            now = time.monotonic()
            if pending and writer is None and now >= deadline:
                label, kind = pending
                review, review_sha = (read_pose_review(ROOT / config['pose_review_file'], args.mode, label)
                                      if kind == 'positive' else (None, None))
                writer = TrialWriter(output, config, args.mode, args.participant_id, args.session_id,
                                     label, kind, context, review, review_sha)
                pending = None
                record_started, last_saved = now, -float('inf')
            if writer is not None:
                elapsed = now-record_started
                if elapsed >= config['capture_seconds']:
                    finished = writer.finish(completed=True, reason='timer_complete')
                    if finished['status'] == 'completed':
                        counts[finished['label']] += 1
                    print(f"{finished['label']}: {finished['status']}, {finished['frame_count']} frames; "
                          f"session trials={counts[finished['label']]}", flush=True)
                    writer = None
                elif now-last_saved >= config['frame_interval_seconds']:
                    encoded_ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config['jpeg_quality']])
                    if not encoded_ok:
                        raise OSError('Could not encode camera image')
                    writer.add_frame(encoded.tobytes(), elapsed)
                    last_saved = now

            preview = cv2.flip(frame, 1) if config['preview_mirrored'] else frame.copy()
            if writer:
                status = f"REKAM {writer.metadata['label']} | sisa {max(0, config['capture_seconds']-(now-record_started)):.1f}s | frame {writer.frames}"
            elif pending:
                status = f'SIAP {pending[0]} | mulai dalam {max(0, deadline-now):.1f}s'
            else:
                status = 'NETRAL / TIDAK MENYIMPAN - kembali netral sebelum trial berikutnya'
            coverage = ' '.join(f'{label}:{counts[label]}/{config["positive_trials_per_class_per_session"]}'
                                for label in config['languages'][args.mode]['classes'])
            lines = [f'{args.mode.upper()} | {args.participant_id}/{args.session_id} | split {role}',
                     status, coverage, 'Huruf: pose | 1-4: negatif | Space batal | Esc keluar',
                     '1 tanpa tangan; 2 netral; 3 non-isyarat; 4 transisi']
            for index, line in enumerate(lines):
                y = 26+index*26
                cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Pilot webcam SIBI dan BISINDO', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == 32:
                if writer:
                    writer.finish(completed=False, reason='operator_cancel')
                    writer = None
                pending = None
            elif writer is None and pending is None:
                letter, char = chr(key).upper(), chr(key)
                if letter in config['languages'][args.mode]['classes']:
                    review, _ = read_pose_review(ROOT / config['pose_review_file'], args.mode, letter)
                    if review['status'] == 'rejected':
                        print(f'{letter} is rejected in the pose review; revise the protocol before capture.')
                        continue
                    pending, deadline = (letter, 'positive'), now+config['countdown_seconds']
                elif char in config['negative_keys']:
                    pending, deadline = (config['negative_keys'][char], 'negative'), now+config['countdown_seconds']
    finally:
        if writer:
            writer.finish(completed=False, reason='capture_interrupted')
        camera.release()
        cv2.destroyAllWindows()
        if protocol_locked and session_dir.exists():
            summary = {'mode': args.mode, 'participant_id': args.participant_id, 'session_id': args.session_id,
                       'split': role, 'last_capture_started_at_utc': session_started,
                       'completed_trials_by_label': dict(completed_counts(session_dir)),
                       'note': 'Intended labels pending review; images from one trial must stay in one split.'}
            (session_dir / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
            print(f'Session: {session_dir}')


if __name__ == '__main__':
    main()
