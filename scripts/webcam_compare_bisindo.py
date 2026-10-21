"""Compare BISINDO weights and recovery on the same camera frame, without recording."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def compare_frame(rgb, old_model, new_model, baseline_detector, recovery_detector):
    baseline = baseline_detector.extract(rgb, 'bisindo', input_mirrored=False)
    recovered = recovery_detector.extract(rgb, 'bisindo', baseline=baseline)
    predictions = {}
    for name, model, result in [('old_baseline', old_model, baseline),
                                ('new_baseline_diagnostic', new_model, baseline),
                                ('new_recovery', new_model, recovered)]:
        predictions[name] = model.predict(result.raw, result.height/result.width) if result.valid else None
    return baseline, recovered, predictions


def load_comparison(old_run, new_run):
    from sibi_bisindo.coordinate_inference import CoordinatePredictor
    from sibi_bisindo.coordinate_detector import build_detector
    old_model, new_model = CoordinatePredictor(old_run), CoordinatePredictor(new_run)
    old_data = json.loads((old_run/'dataset.json').read_text())
    new_data = json.loads((new_run/'dataset.json').read_text())
    if (old_model.selection['language'] != 'bisindo' or new_model.selection['language'] != 'bisindo'
            or old_model.classes != new_model.classes or old_data.get('runtime_preprocessing')
            or not new_data.get('runtime_preprocessing')):
        raise ValueError('Need baseline BISINDO old run and recovered BISINDO new run, with the same labels')
    old_lock, new_lock = (d['extraction_metadata']['train'] for d in (old_data, new_data))
    if any(old_lock[key] != new_lock[key] for key in ('config_sha256', 'feature_implementation_sha256')):
        raise ValueError('Baseline detector configurations differ; shared-frame comparison would be invalid')
    baseline = build_detector(old_data, old_model.selection, ROOT, old_run)
    try:
        recovery = build_detector(new_data, new_model.selection, ROOT, new_run)
    except Exception:
        baseline.close()
        raise
    return old_model, new_model, baseline, recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-run', type=Path, default=ROOT/'artifacts/bisindo-transfer-v1/bisindo_20261020_152850_281288')
    parser.add_argument('--new-run', type=Path, default=ROOT/'artifacts/bisindo-transfer-recovery-v2/bisindo_20261021_133454_756637')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
    from webcam_sibi_classifier import draw_landmarks

    torch.set_num_threads(2)
    old_model, new_model, baseline, recovery = load_comparison(args.old_run, args.new_run)
    camera = None
    try:
        print(f'Old: {args.old_run}')
        print(f'New: {args.new_run}')
        print('Same RGB frame and same baseline landmarks for both classifiers. No recording.')
        print('NEW/base is a diagnostic preprocessing ablation; it is not the trained v2 inference policy.')
        if args.check_only:
            print('Comparison checks passed; camera was not opened.')
            return
        camera = cv2.VideoCapture(args.device, cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY)
        if not camera.isOpened(): raise SystemExit('Cannot open camera; try --device 1')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ok, bgr = camera.read()
            if not ok: break
            started = time.perf_counter()
            base_result, new_result, predictions = compare_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                old_model, new_model, baseline, recovery)
            canvas = np.full((610,960,3), 24, dtype=np.uint8)
            for column, result in enumerate((base_result, new_result)):
                frame = bgr.copy()
                draw_landmarks(frame, result.raw, mp.solutions.hands.HAND_CONNECTIONS, cv2)
                canvas[45:405,column*480:(column+1)*480] = cv2.resize(cv2.flip(frame, 1),(480,360))
            def line(text, y, x=12, color=(245,245,245)):
                cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            line(f'Baseline: {len(base_result.raw)} hand(s)',28)
            line(f'Recovery: {len(new_result.raw)} hand(s)',28,492)
            def label(name):
                p=predictions[name]
                return 'no coordinates' if p is None else f"{p['label']} | score {p['score']:.3f}"
            line('OLD / baseline: '+label('old_baseline'),435)
            line('NEW / same baseline (diagnostic): '+label('new_baseline_diagnostic'),463)
            line('NEW / recovery (normal v2): '+label('new_recovery'),491)
            details = recovery.last_details
            line(f"Recovery {details['method']} | views {details['attempts']} | total {1000*(time.perf_counter()-started):.0f} ms",523)
            line('Compare letters on the SAME pose. Scores are uncalibrated.',555)
            line('q/Esc exits. Preview mirrored; inference unmirrored. Nothing saved.',585)
            cv2.imshow('BISINDO comparison',canvas)
            if cv2.waitKey(1) & 0xff in (27,ord('q')): break
    finally:
        if camera is not None: camera.release()
        baseline.close(); recovery.close(); cv2.destroyAllWindows()


if __name__ == '__main__': main()
