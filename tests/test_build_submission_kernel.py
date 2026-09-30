import ast
import json

import pytest

from scripts import build_submission_kernel as bk

Q, D = chr(39), chr(34)   # single and double quote


def args(tmp_path, stage, **kw):
    base = dict(owner="team", stage=stage, out=tmp_path / stage, shard=None, shards=4, tier="accuracy",
                allow_dirty=True)
    return bk.argparse.Namespace(**{**base, **kw})


def built(tmp_path, stage, **kw):
    out = bk.build(args(tmp_path, stage, **kw))
    return (json.loads((out / "kernel-metadata.json").read_text(encoding="utf-8")),
            (out / "run.py").read_text(encoding="utf-8"),
            json.loads((out / "payload-files.json").read_text(encoding="utf-8")))


def test_every_stage_kernel_is_valid_python_and_names_its_stage(tmp_path):
    for stage, kw in (("documents", {}), ("hands-on", {}), ("queries", {"shard": "1/4"}), ("p0", {})):
        meta, code, _ = built(tmp_path / stage, stage, **kw)
        ast.parse(code)
        assert f'{Q}--stage{Q}, {Q}{stage}{Q}' in code and meta["kernel_type"] == "script" and meta["is_private"] is True


def test_only_the_documents_stage_gets_a_gpu_and_only_with_the_gpu_wheels(tmp_path):
    docs_meta, docs_code, _ = built(tmp_path / "d", "documents")
    cpu_meta, cpu_code, _ = built(tmp_path / "h", "hands-on")
    assert docs_meta["enable_gpu"] is True and "whl/cu128" in docs_code and (f'os.environ[{D}CUDA_VISIBLE_DEVICES{D}] = {Q}0{Q}') in docs_code
    assert cpu_meta["enable_gpu"] is False and "whl/cpu" in cpu_code and "whl/cu128" not in cpu_code
    assert (f'os.environ[{D}CUDA_VISIBLE_DEVICES{D}] = {Q}{Q}') in cpu_code


def test_kernels_chain_through_kernel_sources(tmp_path):
    assert built(tmp_path / "q", "queries", shard="2/4")[0]["kernel_sources"] == ["team/codedoctor-submission-documents"]
    p0 = built(tmp_path / "p", "p0")[0]
    assert p0["kernel_sources"] == ["team/codedoctor-submission-documents",
                                    *[f"team/codedoctor-submission-queries-{i}-of-4" for i in range(4)]]
    assert built(tmp_path / "h", "hands-on", tier="interactive")[0]["id"].endswith("hands-on-06b")


def test_the_commit_is_baked_in_and_a_dirty_tree_is_refused(tmp_path, monkeypatch):
    meta, code, payload = built(tmp_path / "ok", "hands-on")
    head, _ = bk.git_state()
    assert f'os.environ[{D}CODEDOCTOR_COMMIT{D}] = {Q}{head}{Q}' in code and payload["commit"] == head
    assert {"app/submission.py", "config/submission.yaml", "scripts/submission_run.py"} <= set(payload["files"]) \
        or "app" in payload["files"]
    monkeypatch.setattr(bk, "git_state", lambda: ("deadbeef", True))
    with pytest.raises(SystemExit, match="uncommitted"):
        bk.build(args(tmp_path / "no", "hands-on", allow_dirty=False))
    marked = bk.build(args(tmp_path / "yes", "hands-on", allow_dirty=True))
    assert f'os.environ[{D}CODEDOCTOR_DIRTY{D}] = {Q}1{Q}' in (marked / "run.py").read_text(encoding="utf-8")


def test_the_lock_file_is_preferred_when_it_exists(tmp_path, monkeypatch):
    _, code, payload = built(tmp_path / "a", "hands-on")
    expected = "requirements-lock.txt" if (bk.ROOT / "requirements-lock.txt").exists() else "requirements-submission.txt"
    assert payload["requirements"] == expected and expected in code
