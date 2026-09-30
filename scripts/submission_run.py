"""Recorded entrypoint for final validation, corpus preparation and P0 reporting."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import psutil
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['hands-on','documents','p0'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corpus-cache',type=Path)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    manifest={'stage':a.stage,'status':'started','platform':platform.platform(),'python':sys.version,
              'logical_cpus':os.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'ram_bytes':psutil.virtual_memory().total,
              'source_hashes':{str(f.relative_to(ROOT)):digest(f) for top in ('app','config','scripts') for f in sorted((ROOT/top).rglob('*')) if f.suffix in ('.py','.yaml','.json')},
              'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    if Path('/proc/cpuinfo').exists():(out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
    (out/'environment.json').write_text(json.dumps(manifest,indent=2))
    with (out/'requirements-measured.txt').open('w') as handle:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=handle,check=True)
    if a.stage=='hands-on':command=[sys.executable,'scripts/validate_hands_on.py','--out',str(out/'hands-on')]
    elif a.stage=='documents':command=[sys.executable,'scripts/prepare_document_cache.py','--device','cuda','--out',str(out/'documents')]
    else:
        source=a.corpus_cache
        if source is None:
            candidates=list(Path('/kaggle/input').rglob('corpus.sqlite'))
            if len(candidates)!=1:raise RuntimeError('Provide exactly one verified corpus cache via --corpus-cache')
            source=candidates[0]
        provenance=json.loads(source.with_name('manifest.json').read_text())
        if digest(source)!=provenance['sha256']:raise RuntimeError('Offline cache checksum mismatch')
        if provenance['revision']!='3766d46e7a68545ed6190c15330983f9b39ab718':raise RuntimeError('Offline model revision mismatch')
        cache=out/'corpus.sqlite';shutil.copy2(source,cache)
        (out/'document-provenance.json').write_text(json.dumps(provenance,indent=2))
        command=[sys.executable,'scripts/evaluate_mteb.py','--device','cpu','--config','config/submission.yaml','--corpus-cache',str(cache),'--output',str(out/'appsretrieval_results.json'),'--predictions',str(out/'predictions')]
    manifest['command']=command;(out/'environment.json').write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter();peak=0
    with (out/'run.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        def copy_output():
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
        reader=threading.Thread(target=copy_output,daemon=True);reader.start()
        timeout=43200 if a.stage=='p0' else 10800
        while proc.poll() is None:
            try:
                parent=psutil.Process(proc.pid);peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)]))
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if time.perf_counter()-started>timeout:
                for child in psutil.Process(proc.pid).children(recursive=True):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                proc.kill();break
            time.sleep(.1)
        code=proc.wait();reader.join(timeout=10)
    manifest.update(status='completed' if code==0 else 'failed',exit_code=code,seconds=time.perf_counter()-started,peak_process_tree_rss_bytes=peak)
    (out/'environment.json').write_text(json.dumps(manifest,indent=2))
    if code:raise SystemExit(code)
    if a.stage=='p0':subprocess.run([sys.executable,'scripts/validate_submission.py',str(out/'appsretrieval_results.json')],check=True)

if __name__=='__main__':main()
