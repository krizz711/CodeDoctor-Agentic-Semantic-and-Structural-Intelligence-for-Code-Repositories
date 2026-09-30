"""EXT-dev only: do the 9 crashing golds verify when stdin is a file (not a pipe) and exit/quit exist? Also a negative control: same change on 20 non-gold candidates."""
import json, sys, subprocess, tempfile, os
from pathlib import Path
REPO=Path(r"C:\Users\Asus\manish\CodeDoctor"); sys.path.insert(0,str(REPO))
from scripts.replay_outcomes import load_store
from scripts.dev_eval import load_jsonl_texts
from app.pipeline.execution_rerank import _normalize_output
g={k:next(iter(v)) for k,v in json.loads((REPO/"artifacts/ext_cc/qrels.json").read_text(encoding="utf-8")).items()}
c,t=load_jsonl_texts(REPO/"artifacts/ext_cc/texts_corpus.jsonl"); T=dict(zip(c,t))
h,es=load_store(REPO/"data/outcomes/f2llm17b_extcc_dev_deep50.jsonl.gz"); E={e["qid"]:e for e in es}
PRE="import sys as _s\nexit = quit = _s.exit\n"
def run(code, stdin, mode, prelude):
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"c.py"); open(p,"w",encoding="utf-8").write((PRE if prelude else "")+code)
        env={"PATH":os.environ.get("PATH",""),"SYSTEMROOT":os.environ.get("SYSTEMROOT",""),"PYTHONIOENCODING":"utf-8"}
        try:
            if mode=="file":
                f=os.path.join(d,"in.txt"); open(f,"w",encoding="utf-8",newline="\n").write(stdin)
                with open(f,"rb") as fh: r=subprocess.run([sys.executable,"-I","-S",p],stdin=fh,capture_output=True,timeout=5,cwd=d,env=env)
            else:
                r=subprocess.run([sys.executable,"-I","-S",p],input=stdin.encode(),capture_output=True,timeout=5,cwd=d,env=env)
        except subprocess.TimeoutExpired: return "timeout"
        if r.returncode!=0: return "crash:"+r.stderr.decode(errors="replace").strip().splitlines()[-1][:60]
        return r.stdout.decode(errors="replace")
qs=["cc-valid-87","cc-valid-104","cc-test-120","cc-test-125","cc-test-136","cc-test-156","cc-test-191","cc-test-193","cc-test-222"]
for q in qs:
    e=E[q]; ex=e["stdin_examples"][0]; want=_normalize_output(ex["expected"])
    res=[]
    for mode,pre in [("pipe",False),("file",False),("file",True)]:
        o=run(T[g[q]], ex["stdin"], mode, pre); res.append("MATCH" if _normalize_output(o)==want else (o[:40] if o.startswith(("crash","timeout")) else "mismatch"))
    print(q, "pipe/file/file+exit:", res)
# negative control: non-gold top-10 candidates of these statements under file+exit: count matches vs pipe
m_pipe=m_new=0; n=0
for q in qs:
    e=E[q]; ex=e["stdin_examples"][0]; want=_normalize_output(ex["expected"])
    for d in e["pool"][:10]:
        if d==g[q]: continue
        n+=1; m_pipe+= _normalize_output(run(T[d],ex["stdin"],"pipe",False))==want; m_new+= _normalize_output(run(T[d],ex["stdin"],"file",True))==want
print("non-gold top-10 candidates:",n,"verify under pipe:",m_pipe,"under file+exit:",m_new)
