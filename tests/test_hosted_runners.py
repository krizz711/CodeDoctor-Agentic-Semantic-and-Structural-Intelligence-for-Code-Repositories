"""The hosted-runner generators must ship everything they then invoke.

There are three generators (Kaggle script kernel, Kaggle notebook, Colab
notebook) and one packer behind all of them.  The failure this file exists to
prevent is a quiet one: a generator emits a cell that runs
``scripts/something.py``, the packer does not include that path, and the cell
dies with "can't open file" *after* a three-hour GPU run has already finished.

That is not hypothetical.  ``scripts/validate_submission.py`` was invoked by
the Colab notebook and absent from ``PAYLOAD_PATHS`` until 2026-09-19, so the
run always succeeded and the check on it never once executed.

The tests derive what must be packed by reading the generated artifacts, so a
new cell that calls a new script is covered without anyone remembering to
update a list here.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import re
import tarfile

import pytest

from app.packaging import PAYLOAD_PATHS, build_payload
from scripts.make_kaggle_notebook import build_notebook as build_kaggle_notebook
from scripts.make_notebook import build_notebook as build_colab_notebook

#: ``!python scripts/foo.py`` or ``python scripts/foo.py`` inside a cell.
SCRIPT_CALL = re.compile(r"scripts/[A-Za-z0-9_]+\.py")
#: Data files a cell passes as an argument.
DATA_ARG = re.compile(r"(?<![\w/])data/[A-Za-z0-9_]+\.json")
MAGIC = re.compile(r"\s*[!%]")


@pytest.fixture(scope="module")
def payload_paths() -> list[str]:
    _, included = build_payload()
    return included


@pytest.fixture(scope="module")
def notebooks() -> dict[str, dict]:
    # A placeholder payload keeps these fast: the tests below read the cells,
    # never the tarball.
    return {
        "kaggle": build_kaggle_notebook("", tier="accuracy", execution_candidates=20),
        "colab": build_colab_notebook(""),
    }


def code_cells(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


class TestPayloadCoversWhatTheCellsInvoke:
    @pytest.mark.parametrize("name", ["kaggle", "colab"])
    def test_every_invoked_script_is_packed(self, name, notebooks, payload_paths):
        invoked = {
            match
            for source in code_cells(notebooks[name])
            for match in SCRIPT_CALL.findall(source)
        }
        assert invoked, f"{name}: found no script invocations to check"
        missing = sorted(invoked - set(payload_paths))
        assert not missing, (
            f"{name} notebook invokes {missing}, which build_payload() does not "
            f"ship. The cell will fail with 'can't open file' after the run."
        )

    @pytest.mark.parametrize("name", ["kaggle", "colab"])
    def test_every_referenced_data_file_is_packed(self, name, notebooks, payload_paths):
        referenced = {
            match
            for source in code_cells(notebooks[name])
            for match in DATA_ARG.findall(source)
        }
        missing = sorted(referenced - set(payload_paths))
        assert not missing, f"{name} notebook reads {missing}, which is not packed"

    def test_validate_submission_is_packed(self, payload_paths):
        """Named explicitly: this is the path that was actually broken."""
        assert "scripts/validate_submission.py" in payload_paths

    def test_declared_paths_all_exist_on_disk(self):
        _, included = build_payload()
        missing = sorted(set(PAYLOAD_PATHS) - set(included))
        assert not missing, f"PAYLOAD_PATHS names paths that do not exist: {missing}"


class TestPayloadContents:
    def test_tarball_unpacks_and_carries_the_pipeline(self, tmp_path):
        payload, _ = build_payload()
        with tarfile.open(
            fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz"
        ) as tar:
            # `filter` landed in 3.11.4; the hosted runners are on 3.11 and call
            # plain extractall, so tolerate both here.
            try:
                tar.extractall(tmp_path, filter="data")
            except TypeError:  # pragma: no cover - older interpreters only
                tar.extractall(tmp_path)
        assert (tmp_path / "app" / "pipeline" / "encoder.py").exists()
        assert (tmp_path / "config" / "frozen.yaml").exists()
        assert (tmp_path / "scripts" / "validate_submission.py").exists()

    def test_no_bytecode_is_shipped(self):
        payload, _ = build_payload()
        with tarfile.open(
            fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz"
        ) as tar:
            names = tar.getnames()
        assert not [n for n in names if n.endswith(".pyc") or "__pycache__" in n]


class TestGeneratedNotebooksAreWellFormed:
    @pytest.mark.parametrize("name", ["kaggle", "colab"])
    def test_nbformat_shape(self, name, notebooks):
        notebook = notebooks[name]
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        for index, cell in enumerate(notebook["cells"]):
            assert cell["cell_type"] in {"code", "markdown"}, index
            source = cell["source"]
            assert isinstance(source, list) and source, f"cell {index} empty"
            for line in source[:-1]:
                assert line.endswith("\n"), f"cell {index}: unterminated line {line!r}"

    @pytest.mark.parametrize("name", ["kaggle", "colab"])
    def test_code_cells_are_syntactically_valid_python(self, name, notebooks):
        for index, source in enumerate(code_cells(notebooks[name])):
            kept, continuing = [], False
            for line in source.split("\n"):
                if MAGIC.match(line) or continuing:
                    continuing = line.rstrip().endswith("\\")
                    kept.append("")
                    continue
                kept.append(line)
            try:
                ast.parse("\n".join(kept))
            except SyntaxError as exc:
                pytest.fail(f"{name} code cell {index}: {exc.msg} on line {exc.lineno}")

    def test_kaggle_notebook_is_serialisable(self):
        notebook = build_kaggle_notebook("PAYLOAD", tier="accuracy", execution_candidates=20)
        assert json.loads(json.dumps(notebook)) == notebook

    def test_kaggle_notebook_states_the_depth_it_was_built_for(self):
        notebook = build_kaggle_notebook("", tier="accuracy", execution_candidates=20)
        joined = "".join(code_cells(notebook))
        assert "EXECUTION_CANDIDATES = 20" in joined

    def test_kaggle_notebook_warns_that_it_executes_dataset_code(self, notebooks):
        prose = "".join(
            "".join(cell["source"])
            for cell in notebooks["kaggle"]["cells"]
            if cell["cell_type"] == "markdown"
        ).lower()
        assert "executes python that came from the dataset" in prose
