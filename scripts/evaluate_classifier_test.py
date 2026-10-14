"""Evaluate a frozen classifier selection once on the locked test split.

Run only after reviewing validation and accepting selection.json. This script
never retrains or tunes; output defaults to artifacts/classification-v1.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
from sibi_bisindo.training import evaluate_predictions
from train_baselines import read_split,write_json

OUT=ROOT/'artifacts/classification-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages',nargs='+',choices=['sibi','bisindo'],default=['sibi','bisindo'])
    parser.add_argument('--features-dir',type=Path,default=ROOT/'data/features/landmarks-v1-final')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--confirm-frozen-selection',action='store_true',help='Acknowledge that validation selection is final before test access.')
    args=parser.parse_args()
    if not args.confirm_frozen_selection:parser.error('Pass --confirm-frozen-selection only after validation-based selection is final.')
    features_dir=args.features_dir.resolve();output_dir=args.output_dir.resolve()
    for language in args.languages:
        selection_path=output_dir/language/'selection.json'
        selection=json.loads(selection_path.read_text(encoding='utf-8'))
        if selection['language']!=language or selection['test_accessed'] is not False:
            raise ValueError(f'Invalid selection lock for {language}')
        report_path=output_dir/language/'test_metrics.json'
        if report_path.exists():
            raise ValueError(f'Test already evaluated for {language}; do not tune against this test set')
        artifact=ROOT/selection['artifact']
        if sha(artifact)!=selection['artifact_sha256']:
            raise ValueError(f'Selected model artifact changed: {artifact}')
        if str(features_dir.relative_to(ROOT))!=selection['features_dir']:
            raise ValueError('Feature cache differs from selected validation run')
        feature_lock=json.loads((features_dir/'extraction_lock.json').read_text(encoding='utf-8'))
        if feature_lock['config_sha256']!=selection['feature_config_sha256']:
            raise ValueError('Feature configuration differs from validation selection')
        selection['test_accessed']=True
        write_json(selection_path,selection)
        x,y,mask,paths,metadata=read_split(features_dir/f'{language}_test.npz')
        if metadata['split']!='test' or metadata['language']!=language:
            raise ValueError('Test cache identity mismatch')
        if metadata['config_sha256']!=selection['feature_config_sha256']:
            raise ValueError('Test feature identity differs from frozen validation selection')
        bundle=joblib.load(artifact);model=bundle['model']
        xt=x[mask]
        if bundle['kind']=='mlp':xt=bundle['scaler'].transform(xt).astype(np.float32)
        pred=np.full(len(y),'',dtype='<U2');pred[mask]=model.predict(xt)
        metrics=evaluate_predictions(y,pred,mask,selection['classes'])
        result={'version':'classifier-test-report-v1','language':language,
            'selected_candidate':selection['selected_candidate'],'artifact_sha256':selection['artifact_sha256'],
            'test_manifest_sha256':metadata['manifest_sha256'],'metrics':metrics,
            'limitations':selection['limitations']+['This is a one-time frozen test report; do not tune from it.']}
        write_json(report_path,result)
        print(f"[{language}] test system macro-F1={metrics['system_all_images']['macro_f1']:.4f}; valid-only macro-F1={metrics['classifier_valid_only']['macro_f1']:.4f}; coverage={metrics['feature_coverage']:.2%}",flush=True)


if __name__=='__main__':main()
