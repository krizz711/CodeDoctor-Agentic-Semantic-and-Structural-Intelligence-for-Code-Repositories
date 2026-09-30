"""Final P1/Bonus functional measurements through the shipping workbench/index."""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.indexing.cache import EmbeddingCache  # noqa: E402
from app.indexing.versioned import VersionedIndex  # noqa: E402
from app.retrieval.encode import CodeEncoder  # noqa: E402
from app.retrieval.hybrid import Document  # noqa: E402
from app.submission import execution_config  # noqa: E402
from app.workbench import Workbench, _documents_from_source, _encoder_input, DENSE_MAX_INPUT_CHARS  # noqa: E402


def texts(folder):
    return {_encoder_input(doc) for p in folder.glob('*.py')
            for doc in _documents_from_source(p.read_text(),p.name)
            if len(_encoder_input(doc))<=DENSE_MAX_INPUT_CHARS}


def build(bench,key,rebuild):
    started=time.perf_counter();bench.start_build(key,dense=True,rebuild=rebuild)
    while time.perf_counter()-started<1800:
        state=bench.state()
        if state['phase']=='error':raise RuntimeError(state)
        if state['phase']=='ready':
            return {'seconds':time.perf_counter()-started,**{k:state.get(k) for k in ('encoded','reused','chunks')}}
        time.sleep(.05)
    raise TimeoutError('Index operation exceeded 30 minutes')


def fixture(folder, values):
    for i,value in enumerate(values):
        (folder/f'f{i}.py').write_text(f'def add_{i}(value):\n    """Add {value} to an integer."""\n    return value + {value}\n')


def p1(encoder,work):
    runs=[]
    for repetition in range(3):
        folder=work/f'repo{repetition}';folder.mkdir();values=list(range(40));fixture(folder,values)
        bench=Workbench(cache=EmbeddingCache(work/f'cache{repetition}.db'));bench._encoder=encoder
        key=bench.open_folder(str(folder))['key'];seen=texts(folder)
        record={'repetition':repetition,'full':build(bench,key,False),'updates':[]}
        for operation in ('unchanged','one','ten','add','delete','rename'):
            if operation=='one':values[0]=101;fixture(folder,values)
            if operation=='ten':values[:10]=list(range(200,210));fixture(folder,values)
            if operation=='add':(folder/'added.py').write_text('def added(x):\n    return x * 71\n')
            if operation=='delete':(folder/'added.py').unlink()
            if operation=='rename':(folder/'f39.py').rename(folder/'moved.py')
            current=texts(folder);expected=len(current-seen)
            row={'operation':operation,'expected_new_inputs':expected,**build(bench,key,True)}
            record['updates'].append(row);seen.update(current)
            if row['encoded']!=expected:raise AssertionError(f'cache accounting mismatch: {row}')
        query='Add 205 to an integer'
        started=time.perf_counter();inc=bench.search(key,query,verify=False);record['query_seconds']=time.perf_counter()-started
        fresh=Workbench(cache=EmbeddingCache(work/f'fresh{repetition}.db'));fresh._encoder=encoder
        fresh_key=fresh.open_folder(str(folder))['key'];record['fresh']=build(fresh,fresh_key,False)
        rebuilt=fresh.search(fresh_key,query,verify=False)
        def identities(result):return [(r.get('file_path'),r.get('name')) for r in result['results']]
        record['fresh_ranking_equal']=identities(inc)==identities(rebuilt)
        if not record['fresh_ranking_equal']:raise AssertionError('fresh/incremental ranking differs')
        runs.append(record)
    return {'fixture':'40 Python functions; three repetitions','runs':runs}


def bonus(encoder):
    index=VersionedIndex(encoder=encoder);correct={};lineages={};vectors={}
    for version in range(3):
        docs=[]
        for i in range(12):
            # Alternate where the correct code occurs. No recency shortcut.
            delta=i+1 if version==i%3 else i+101
            code=f'n = int(input())\nprint(n + {delta})\n'
            name=f'add_{i+1}';doc=Document(document_id=f'{i}-{version}',file_path=f'{name}.py',content=code,name=name,type='FUNCTION')
            docs.append(doc);lineages[code]=name
            if version==i%3:correct[i]=code
            if code not in vectors:vectors[code]=encoder.encode([code],mode='document',batch_size=1)[0]
        index.add_snapshot(str(version),docs,embeddings=np.asarray([vectors[d.content] for d in docs]))
    rows=[]
    for i in range(12):
        query=f'Add {i+1} to the input integer.\n\n-----Example-----\nInput\n3\n\nOutput\n{i+4}\n\n-----Example-----\nInput\n8\n\nOutput\n{i+9}\n'
        started=time.perf_counter()
        grouped,raw=index.retrieve_across_versions(query,limit=10,execution=execution_config(),return_ungrouped=True)
        elapsed=time.perf_counter()-started;gold=correct[i];family=f'add_{i+1}'
        rank=next((j for j,h in enumerate(raw) if h.document.content==gold),len(raw)+1)
        rows.append({'query_id':i,'seconds':elapsed,'correct_version_first':bool(grouped and grouped[0].document.content==gold),
                     'lineage_first':bool(grouped and grouped[0].document.name==family),
                     'buggy_above':any(j<rank and h.document.name==family and h.document.content!=gold for j,h in enumerate(raw)),
                     'distinct_at_10':len({h.document.name for h in grouped}),
                     'ranking':[{'code':h.document.content,'name':h.document.name,'versions':h.versions,'sources':h.sources} for h in grouped]})
    metrics={key:float(np.mean([r[key] for r in rows])) for key in ('seconds','correct_version_first','lineage_first','buggy_above','distinct_at_10')}
    return {'kind':'functional synthetic diagnostic, not generalization evidence','metrics':metrics,'queries':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    encoder=CodeEncoder(tier='accuracy',device='cpu',torch_dtype='float32')
    started=time.perf_counter();smoke=encoder.encode(['Return the sum of two integers.'],mode='query',batch_size=1)
    assert smoke.shape==(1,2048) and np.isfinite(smoke).all()
    metadata={'model':encoder.model_name,'revision':encoder.revision,'platform':platform.platform(),'threads':os.cpu_count(),'cold_smoke_seconds':time.perf_counter()-started}
    (args.out/'environment.json').write_text(json.dumps(metadata,indent=2))
    with tempfile.TemporaryDirectory(prefix='codedoctor-validation-') as folder:
        result=p1(encoder,Path(folder));(args.out/'p1.json').write_text(json.dumps(result,indent=2))
    result=bonus(encoder);(args.out/'bonus.json').write_text(json.dumps(result,indent=2))
    print('Hands-on validation complete',flush=True)

if __name__=='__main__':main()
