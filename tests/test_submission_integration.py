"""Judge-facing integration: real verification and version evidence, no model download."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from app.submission import execution_config, SUBMISSION_CONFIG
from app.retrieval.encode import CodeEncoder
from app.retrieval.hybrid import Document, RankedDocument
from app.indexing.versioned import VersionedIndex
from app.pipeline.encoder import PrePostPipelineEncoder
from app.workbench import Workbench

QUERY = 'Print the sum.\n\n-----Example-----\nInput\n2 3\n\nOutput\n5\n'
GOOD = 'a,b=map(int,input().split()); print(a+b)'
BAD = 'a,b=map(int,input().split()); print(a-b)'

def doc(code, name='solve'):
    return Document(document_id=code, file_path='solve.py', content=code, name=name, type='FUNCTION')

def test_accuracy_and_accepted_settings_are_default():
    encoder=CodeEncoder()
    assert encoder.tier=='accuracy' and encoder.config_path==SUBMISSION_CONFIG
    assert encoder.model_name.endswith('1.7B')
    config=execution_config()
    assert config.call_style and config.blank_line_fallback and config.judge_like_io
    assert config.skip_interactive and config.deepen_to==50 and config.timeout_mode=='cpu'

def test_workbench_and_scored_path_agree_on_deepening():
    config=replace(execution_config(),max_candidates=1,deepen_to=2,workers=1)
    hits=[RankedDocument(doc(BAD),.9,('dense',)),RankedDocument(doc(GOOD),.8,('dense',))]
    built=SimpleNamespace(target=SimpleNamespace(language='Python'))
    result=Workbench()._verify(QUERY,hits,built,enabled=True,config=config)
    pipeline=PrePostPipelineEncoder(object(),execution_config=config)
    pipeline._query_texts=[QUERY];pipeline._document_texts=[BAD,GOOD]
    scores=pipeline._apply_execution(np.array([[.9,.8]],dtype=np.float32))[0]
    assert np.argsort(-scores).tolist()==np.argsort(-np.array(result['adjusted_scores'])).tolist()==[1,0]
    assert result['per_result'][1]['state']=='verified'
    assert result['executed']==2

def test_cross_version_verification_selects_correct_content_and_deduplicates(monkeypatch):
    import app.pipeline.execution_rerank as er
    index=VersionedIndex()
    for version,code in [('one',BAD),('two',GOOD),('three',GOOD)]:
        index.add_snapshot(version,[doc(code)])
    calls=[];original=er.verify_candidate
    def observed(code,*args,**kwargs):
        calls.append(code);return original(code,*args,**kwargs)
    monkeypatch.setattr(er,'verify_candidate',observed)
    grouped,raw=index.retrieve_across_versions(QUERY,execution=replace(execution_config(),workers=1),return_ungrouped=True)
    assert grouped[0].document.content==GOOD
    assert grouped[0].versions==['one','two','three']
    assert grouped[0].variants==2 and 'execution' in grouped[0].sources
    assert len(calls)==2 and len(raw)==3

def test_no_examples_keeps_cross_version_ranking():
    index=VersionedIndex()
    index.add_snapshot('one',[doc(BAD)])
    index.add_snapshot('two',[doc(GOOD)])
    before=index.retrieve_across_versions('solve')
    after=index.retrieve_across_versions('solve',execution=execution_config())
    assert [(h.content_hash,h.score) for h in before]==[(h.content_hash,h.score) for h in after]
