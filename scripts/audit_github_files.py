"""Audit Git-tracked and optionally untracked publishable files without staging."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_ROOTS={'dataset','data','artifacts','models','checkpoints','runs','recordings','uploads','outputs','.deploy'}
PRIVATE_SUFFIXES={'.pt','.pth','.onnx','.npy','.npz','.joblib','.pkl','.pickle','.h5','.keras','.tflite','.task',
                  '.zip','.tar','.gz','.tgz','.pem','.key','.p12','.pfx','.mp4','.avi','.mov','.jpg','.jpeg','.png'}


def reason(name):
    path=Path(name);parts=path.parts
    if any(p.startswith('.venv') or p in {'__pycache__','.aws'} for p in parts):return 'environment/cache/private directory'
    if parts and parts[0] in PRIVATE_ROOTS:return 'local data or model directory'
    if path.name.startswith('.env') and not path.name.endswith('.example'):return 'environment secrets'
    if path.suffix.lower() in PRIVATE_SUFFIXES:return 'model/data/archive/media/private-key file'
    if path.name.startswith(('id_rsa','id_ed25519')):return 'SSH key'
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--include-untracked',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    command=['git',f'--git-dir={ROOT/".git"}',f'--work-tree={ROOT}','ls-files','-z','--cached']
    if args.include_untracked:command+=['--others','--exclude-standard']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,check=True)
    files=sorted(set(x for x in result.stdout.decode('utf-8').split('\0') if x));issues=[]
    for name in files:
        problem=reason(name);path=ROOT/name
        if problem:issues.append({'path':name,'reason':problem});continue
        if not path.is_file():continue
        if path.stat().st_size>10*1024*1024:issues.append({'path':name,'reason':'over 10 MiB; review before publishing'});continue
        if path.suffix=='.ipynb':
            notebook=json.loads(path.read_text(encoding='utf-8'))
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook.get('cells',[]) if c['cell_type']=='code'):
                issues.append({'path':name,'reason':'executed notebook outputs; clear before committing'})
        if path.suffix in {'.py','.md','.json','.yml','.yaml','.txt','.example'}:
            text=path.read_text(encoding='utf-8-sig')
            if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}',text):
                issues.append({'path':name,'reason':'possible credential content'})
    report={'files_checked':len(files),'issues':issues,'scope':'tracked and untracked' if args.include_untracked else 'tracked'}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if args.check and issues:raise SystemExit(1)


if __name__=='__main__':main()
