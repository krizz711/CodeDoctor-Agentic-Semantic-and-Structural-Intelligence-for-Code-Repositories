"""Pipe vs file stdin for the fast-IO idiom, and exit/quit under -S. Windows, this machine."""
import os, subprocess, sys, tempfile
progs={
 "fstat_read":"import os\ndata=os.read(0, os.fstat(0).st_size)\nprint(len(data), sum(map(int,data.split())))\n",
 "fstat_size_only":"import os\nprint('st_size', os.fstat(0).st_size)\n",
 "bytesio_idiom":"import io,os\ninput=io.BytesIO(os.read(0,os.fstat(0).st_size)).readline\nn=int(input())\nprint(n*2)\n",
 "buffer_read":"import sys\nd=sys.stdin.buffer.read()\nprint(len(d), sum(map(int,d.split())))\n",
 "bare_exit":"n=int(input())\nprint(n)\nexit()\nprint('never')\n",
 "bare_quit_code":"print(1)\nquit(0)\n",
 "exit_str":"print(1)\nexit('bye')\n",
 "exit_early_partial":"a=int(input())\nif a>0:\n    print('first')\n    exit()\nprint('second')\n",
}
stdin_text="3\n1 2 3\n"
PRE="import sys as _s\nexit = quit = _s.exit\n"
def run(code, mode, prelude, flags):
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"c.py"); open(p,"w",encoding="utf-8").write((PRE if prelude else "")+code)
        env={"PATH":os.environ.get("PATH",""),"SYSTEMROOT":os.environ.get("SYSTEMROOT",""),"PYTHONIOENCODING":"utf-8"}
        if mode=="file":
            f=os.path.join(d,"in.txt"); open(f,"w",encoding="utf-8",newline="\n").write(stdin_text)
            with open(f,"rb") as fh: r=subprocess.run([sys.executable,*flags,p],stdin=fh,capture_output=True,timeout=10,cwd=d,env=env)
        else:
            r=subprocess.run([sys.executable,*flags,p],input=stdin_text.encode(),capture_output=True,timeout=10,cwd=d,env=env)
        err=r.stderr.decode(errors="replace").strip().splitlines()
        return f"rc={r.returncode} out={r.stdout.decode(errors='replace').strip()!r} err={err[-1][:60] if err else ''!r}"
print(sys.platform, sys.version.split()[0])
for name,code in progs.items():
    for mode,pre,flags in [("pipe",False,["-I","-S"]),("file",False,["-I","-S"]),("file",True,["-I","-S"]),("pipe",False,["-I"])]:
        print(f"{name:20s} {mode:4s} prelude={pre!s:5s} flags={' '.join(flags):6s} -> {run(code,mode,pre,flags)}")
