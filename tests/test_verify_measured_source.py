import hashlib
import json
import subprocess
import sys

import pytest

from scripts import build_submission_kernel as bk
from scripts import submission_run as run
from scripts import verify_measured_source as vm


def committed_environment(commit, paths=("app/submission.py", "config/submission.yaml", "scripts/submission_run.py")):
    return {"commit": commit,
            "source_sha256": {p: vm.blob_sha256(commit, p) for p in paths},
            "configuration": {"sha256": vm.blob_sha256(commit, "config/submission.yaml")}}


def test_a_run_that_recorded_the_committed_bytes_verifies():
    head, _ = bk.git_state()
    report = vm.verify(committed_environment(head), head)
    assert report["identical"] and report["files_checked"] == 3 and report["commit"] == head


def test_a_changed_or_missing_file_or_configuration_is_reported():
    head, _ = bk.git_state()
    environment = committed_environment(head)
    environment["source_sha256"]["app/submission.py"] = "0" * 64
    environment["source_sha256"]["scripts/does_not_exist.py"] = "1" * 64
    environment["configuration"]["sha256"] = "2" * 64
    report = vm.verify(environment, head)
    assert not report["identical"]
    assert report["differing"] == ["app/submission.py"] and report["missing"] == ["scripts/does_not_exist.py"]
    assert report["configuration_matches"] is False


def test_an_environment_without_hashes_is_an_error():
    with pytest.raises(ValueError):
        vm.verify({"commit": "x"}, "HEAD")


def test_the_kernel_payload_reproduces_the_hashes_a_run_would_record(tmp_path):
    """What a kernel extracts hashes exactly like the committed blobs, whatever the checkout's line endings."""
    import base64
    import io
    import tarfile

    head, _ = bk.git_state()
    payload, files = bk.git_payload(head)
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz").extractall(tmp_path, filter="data")
    hashes = run.source_hashes(tmp_path)
    assert set(hashes) <= set(files)
    for path, digest in hashes.items():
        assert digest == vm.blob_sha256(head, path)
    environment = {"commit": head, "source_sha256": hashes,
                   "configuration": {"sha256": hashes["config/submission.yaml"]}}
    assert vm.verify(environment, head)["identical"]


def test_the_cli_exits_nonzero_when_the_source_differs(tmp_path):
    head, _ = bk.git_state()
    environment = committed_environment(head)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(environment), encoding="utf-8")
    environment["source_sha256"]["app/submission.py"] = hashlib.sha256(b"tampered").hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(environment), encoding="utf-8")
    command = [sys.executable, str(vm.ROOT / "scripts" / "verify_measured_source.py"), "--commit", head]
    assert subprocess.run([*command, "--environment", str(good)], capture_output=True).returncode == 0
    assert subprocess.run([*command, "--environment", str(bad)], capture_output=True).returncode == 1
