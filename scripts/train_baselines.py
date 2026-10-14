"""Compare SVM candidates and train monitored MLP baselines on train/validation only.

Example: python scripts/train_baselines.py --languages sibi bisindo
MLP writes a CSV row and checkpoint every epoch. Test data is never loaded here.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sibi_bisindo.training import evaluate_predictions

DEFAULT_FEATURES=ROOT/'data/features/landmarks-v1-final'
OUT=ROOT/'artifacts/classification-v1'
SEED=42
MLP_EPOCHS=80
PATIENCE=12


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_split(path):
    with np.load(path,allow_pickle=False) as saved:
        x=saved['X'].astype(np.float32)
        y=saved['labels'].astype(str)
        mask=saved['valid_mask'].astype(bool)
        paths=saved['paths'].astype(str)
        metadata=json.loads(str(saved['metadata_json']))
    if not mask.any() or not np.isfinite(x[mask]).all() or not np.isnan(x[~mask]).all():
        raise ValueError(f'Unexpected feature-mask representation: {path}')
    return x,y,mask,paths,metadata


def macro_f1(y,pred,classes):
    from sklearn.metrics import f1_score
    return float(f1_score(y,pred,labels=classes,average='macro',zero_division=0))


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    temp.replace(path)


def train_svm(language,x_train,y_train,train_mask,x_val,y_val,val_mask,classes,run_dir):
    rows=[];best=None
    for kernel in ['linear','rbf']:
        for c_value in [0.1,1.0,10.0]:
            name=f'svm_{kernel}_c{c_value:g}'
            model=make_pipeline(StandardScaler(),SVC(kernel=kernel,C=c_value,
                class_weight='balanced',decision_function_shape='ovr',cache_size=512))
            started=time.perf_counter()
            model.fit(x_train[train_mask],y_train[train_mask])
            pred=model.predict(x_val[val_mask])
            valid_score=macro_f1(y_val[val_mask],pred,classes)
            full_pred=np.full(len(y_val),'',dtype='<U2');full_pred[val_mask]=pred
            metrics=evaluate_predictions(y_val,full_pred,val_mask,classes)
            fit_seconds=time.perf_counter()-started
            artifact=run_dir/f'{name}.joblib'
            joblib.dump({'model':model,'classes':classes,'language':language,'kind':'svm',
                'features_dir':str(DEFAULT_FEATURES),'train_count':int(train_mask.sum()),
                'validation_metrics':metrics},artifact)
            row={'candidate':name,'family':'SVM','validation_valid_macro_f1':valid_score,
                'validation_system_macro_f1':metrics['system_all_images']['macro_f1'],
                'validation_system_accuracy':metrics['system_all_images']['accuracy'],
                'fit_seconds':fit_seconds,'artifact':str(artifact.relative_to(ROOT))}
            rows.append(row)
            print(f"[{language}] {name}: val valid macro-F1={valid_score:.4f}; system macro-F1={row['validation_system_macro_f1']:.4f}; fit={fit_seconds:.1f}s",flush=True)
            if best is None or (row['validation_system_macro_f1'],valid_score)>(best['row']['validation_system_macro_f1'],best['row']['validation_valid_macro_f1']):
                best={'row':row,'artifact':artifact,'metrics':metrics,'kind':'svm','name':name}
    return rows,best


def train_mlp(language,x_train,y_train,train_mask,x_val,y_val,val_mask,classes,run_dir,hidden):
    name='mlp_'+'x'.join(map(str,hidden));scaler=StandardScaler()
    tx=scaler.fit_transform(x_train[train_mask]).astype(np.float32)
    vx=scaler.transform(x_val[val_mask]).astype(np.float32)
    ty=y_train[train_mask];vy=y_val[val_mask]
    model=MLPClassifier(hidden_layer_sizes=tuple(hidden),solver='adam',alpha=1e-4,
        batch_size=128,learning_rate_init=1e-3,shuffle=True,random_state=SEED,
        max_iter=1,warm_start=False,early_stopping=False)
    classes_np=np.asarray(classes,dtype=str)
    rows=[];best_score=-1.;best_epoch=0;best_bundle=None;stale=0
    checkpoint=run_dir/f'{name}_best.joblib';log_path=run_dir/f'{name}_epochs.csv'
    with log_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','train_macro_f1','validation_macro_f1','validation_system_macro_f1','validation_system_accuracy','epoch_seconds','best_epoch','stale_epochs'])
        writer.writeheader()
        for epoch in range(1,MLP_EPOCHS+1):
            started=time.perf_counter()
            model.partial_fit(tx,ty,classes=classes_np if epoch==1 else None)
            train_pred=model.predict(tx);val_pred=model.predict(vx)
            train_score=macro_f1(ty,train_pred,classes)
            val_score=macro_f1(vy,val_pred,classes)
            full_pred=np.full(len(y_val),'',dtype='<U2');full_pred[val_mask]=val_pred
            metrics=evaluate_predictions(y_val,full_pred,val_mask,classes)
            system_score=metrics['system_all_images']['macro_f1']
            elapsed=time.perf_counter()-started
            improved=system_score>best_score+1e-8
            if improved:
                best_score=system_score;best_epoch=epoch;stale=0
                best_bundle={'model':model,'scaler':scaler,'classes':classes,'language':language,
                    'kind':'mlp','features_dir':str(DEFAULT_FEATURES),'train_count':int(train_mask.sum()),
                    'best_epoch':epoch,'validation_metrics':metrics}
                joblib.dump(best_bundle,checkpoint)
            else:stale+=1
            writer.writerow({'epoch':epoch,'train_loss':float(model.loss_),
                'train_macro_f1':train_score,'validation_macro_f1':val_score,
                'validation_system_macro_f1':system_score,
                'validation_system_accuracy':metrics['system_all_images']['accuracy'],
                'epoch_seconds':elapsed,'best_epoch':best_epoch,'stale_epochs':stale})
            f.flush()
            print(f"[{language}] {name} epoch {epoch:03d}/{MLP_EPOCHS}: loss={model.loss_:.5f}; train macro-F1={train_score:.4f}; val macro-F1={val_score:.4f}; system macro-F1={system_score:.4f}; best={best_epoch}; {elapsed:.1f}s",flush=True)
            if stale>=PATIENCE:
                print(f'[{language}] {name}: early stop; best epoch={best_epoch}',flush=True);break
    # The live estimator keeps training after a best epoch; reload its snapshot
    # so reported validation metrics correspond to the saved best checkpoint.
    best_bundle=joblib.load(checkpoint)
    row={'candidate':name,'family':'MLP','validation_valid_macro_f1':macro_f1(vy,best_bundle['model'].predict(vx),classes),
        'validation_system_macro_f1':best_bundle['validation_metrics']['system_all_images']['macro_f1'],
        'validation_system_accuracy':best_bundle['validation_metrics']['system_all_images']['accuracy'],
        'best_epoch':best_epoch,'epochs_run':epoch,'artifact':str(checkpoint.relative_to(ROOT)),
        'epoch_log':str(log_path.relative_to(ROOT))}
    return row,{'row':row,'artifact':checkpoint,'metrics':best_bundle['validation_metrics'],'kind':'mlp','name':name}


def run_language(language,features_dir,output_dir):
    selection_path=output_dir/language/'selection.json'
    if (output_dir/language/'test_metrics.json').exists() or (selection_path.exists() and json.loads(selection_path.read_text(encoding='utf-8')).get('test_accessed')):
        raise RuntimeError(f'Test was already accessed for {language}; start a new data/model version before further tuning')
    x_train,y_train,train_mask,train_paths,train_meta=read_split(features_dir/f'{language}_train.npz')
    x_val,y_val,val_mask,val_paths,val_meta=read_split(features_dir/f'{language}_val.npz')
    if (train_meta['language']!=language or val_meta['language']!=language
            or train_meta['split']!='train' or val_meta['split']!='val'
            or train_meta['config_sha256']!=val_meta['config_sha256']
            or train_meta['feature_implementation_sha256']!=val_meta['feature_implementation_sha256']):
        raise ValueError('Feature-cache language/split identity mismatch')
    classes=sorted(set(y_train))
    if classes!=sorted(set(y_val)):
        raise ValueError(f'Train/validation class mismatch for {language}')
    run_dir=output_dir/language;run_dir.mkdir(parents=True,exist_ok=True)
    print(f'[{language}] train={int(train_mask.sum())}/{len(train_mask)} valid features; validation={int(val_mask.sum())}/{len(val_mask)}; classes={len(classes)}',flush=True)
    if language=='bisindo' and train_mask.mean()<0.8:
        print('[bisindo] WARNING: training feature coverage is below 80%; results are exploratory and may reflect severe split/domain bias.',flush=True)
    results=[]
    svm_rows,svm_best=train_svm(language,x_train,y_train,train_mask,x_val,y_val,val_mask,classes,run_dir)
    results.extend(svm_rows)
    candidates=[svm_best]
    for hidden in [(128,),(128,64)]:
        row,best=train_mlp(language,x_train,y_train,train_mask,x_val,y_val,val_mask,classes,run_dir,hidden)
        results.append(row);candidates.append(best)
    winner=max(candidates,key=lambda c:(c['row']['validation_system_macro_f1'],c['row']['validation_valid_macro_f1']))
    metrics_path=output_dir/language/f"{winner['name']}_validation_metrics.json"
    write_json(metrics_path,winner['metrics'])
    model_sha=sha(winner['artifact'])
    selection={
        'version':'classifier-selection-v1','language':language,
        'selected_candidate':winner['name'],'selected_kind':winner['kind'],
        'artifact':str(winner['artifact'].relative_to(ROOT)),'artifact_sha256':model_sha,
        'selection_metric':'validation system macro-F1 (all images; invalid extraction counted as rejection)',
        'validation_system_macro_f1':winner['row']['validation_system_macro_f1'],
        'validation_valid_only_macro_f1':winner['row']['validation_valid_macro_f1'],
        'features_dir':str(features_dir.relative_to(ROOT)),
        'feature_config_sha256':json.loads((features_dir/'extraction_lock.json').read_text(encoding='utf-8'))['config_sha256'],
        'feature_implementation_sha256':train_meta.get('feature_implementation_sha256'),
        'training_script_sha256':sha(Path(__file__)),
        'scikit_learn_version':sklearn.__version__,'joblib_version':joblib.__version__,
        'dataset_manifest_sha256':train_meta.get('manifest_sha256'),'classes':classes,'seed':SEED,
        'validation_manifest_sha256':val_meta['manifest_sha256'],
        'test_accessed':False,
        'candidates':results,
        'limitations':['Scores are exploratory; BISINDO filename-series split is not verified signer-independent.',
            'BISINDO class/pose and actor assignment are unverified.',
            'Classifier metrics are not webcam/generalization results.',
            'Scores/margins are not calibrated probabilities; no rejection threshold was selected.']}
    write_json(output_dir/language/'selection.json',selection)
    print(f"[{language}] SELECTED {winner['name']} validation system macro-F1={winner['row']['validation_system_macro_f1']:.4f}; selection lock={output_dir/language/'selection.json'}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages',nargs='+',choices=['sibi','bisindo'],default=['sibi','bisindo'])
    parser.add_argument('--features-dir',type=Path,default=DEFAULT_FEATURES)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    features_dir=args.features_dir.resolve();output_dir=args.output_dir.resolve()
    if not (features_dir/'extraction_lock.json').exists():
        raise SystemExit(f'Feature cache lock not found: {features_dir}')
    for language in args.languages:run_language(language,features_dir,output_dir)


if __name__=='__main__':main()
