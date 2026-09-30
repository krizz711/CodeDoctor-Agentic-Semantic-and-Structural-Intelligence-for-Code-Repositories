"""Package the documented submission commands into an isolated Kaggle session."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.packaging import build_payload  # noqa: E402


def build(owner,stage,out):
    paths=('app','config','requirements.txt','requirements-submission.txt','scripts/__init__.py',
           'scripts/evaluate_mteb.py','scripts/export_submission_csv.py','scripts/validate_submission.py',
           'scripts/validate_hands_on.py','scripts/prepare_document_cache.py','scripts/submission_run.py')
    payload,included=build_payload(paths)
    out.mkdir(parents=True,exist_ok=True)
    code=f'''import base64,io,json,os,subprocess,sys,tarfile,time
from pathlib import Path
from urllib.request import urlopen
os.environ['HF_HOME']='/tmp/codedoctor-model-cache'
os.environ['CUDA_VISIBLE_DEVICES']={"'0,1'" if stage=="documents" else "''"}
root=Path('/kaggle/working/source');root.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r})),mode='r:gz') as archive:
    archive.extractall(root,filter='data')
with urlopen('https://pypi.org/simple/',timeout=30) as response:
    print('Internet preflight',response.status,flush=True)
venv=Path('/tmp/codedoctor-env')
subprocess.run([sys.executable,'-m','venv',str(venv)],check=True,timeout=180)
python=str(venv/'bin/python')
subprocess.run([python,'-m','pip','install','--index-url','https://download.pytorch.org/whl/{"cu128" if stage=="documents" else "cpu"}','torch==2.8.0'],check=True,timeout=1200)
subprocess.run([python,'-m','pip','install','--index-url','https://pypi.org/simple','-r',str(root/'requirements-submission.txt')],check=True,timeout=1200)
subprocess.run([python,'scripts/submission_run.py','--stage',{stage!r},'--out','/kaggle/working/results'],cwd=root,check=True,timeout={43200 if stage=="p0" else 10800})
'''
    (out/'run.py').write_text(code)
    meta={'id':f'{owner}/codedoctor-submission-{stage}','title':f'CodeDoctor submission {stage}','code_file':'run.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':stage=='documents','enable_internet':True,'dataset_sources':[],'kernel_sources':([f'{owner}/codedoctor-submission-documents'] if stage=='p0' else [])}
    (out/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    (out/'payload-files.json').write_text(json.dumps(included,indent=2))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner',required=True);p.add_argument('--stage',choices=['hands-on','documents','p0'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.owner,a.stage,a.out)
