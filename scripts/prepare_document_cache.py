"""Offline corpus-only preprocessing; never downloads query or qrel splits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.indexing.cache import EmbeddingCache,recipe_cache_key  # noqa: E402
from app.retrieval.encode import CodeEncoder,select_dtype  # noqa: E402


def main():
    from datasets import load_dataset
    import mteb
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    dataset=dict(mteb.get_task('AppsRetrieval').metadata.dataset)
    if not dataset.get('revision'):raise RuntimeError('MTEB dataset revision is not pinned')
    corpus=load_dataset(dataset['path'],'corpus',split='corpus',revision=dataset['revision'])
    encoder=CodeEncoder(tier='accuracy',device=a.device,torch_dtype=select_dtype(a.device))
    cache=EmbeddingCache(a.out/'corpus.sqlite');started=time.perf_counter()
    texts=list(corpus['text'])
    # Bounded windows retain the encoder's own length-aware batches.
    for start in range(0,len(texts),64):
        batch=texts[start:start+64];vectors=encoder.encode(batch,mode='document',show_progress_bar=False)
        cache.put_many([(recipe_cache_key(encoder,t),v) for t,v in zip(batch,vectors)])
        print('Documents',min(start+64,len(texts)),'/',len(texts),flush=True)
    cache.conn.execute('PRAGMA wal_checkpoint(TRUNCATE)');cache.conn.close()
    manifest={'dataset':dataset,'model':encoder.model_name,'revision':encoder.revision,'device':a.device,'dtype':encoder.torch_dtype,'documents':len(texts),'seconds':time.perf_counter()-started,'sha256':hashlib.sha256((a.out/'corpus.sqlite').read_bytes()).hexdigest()}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
