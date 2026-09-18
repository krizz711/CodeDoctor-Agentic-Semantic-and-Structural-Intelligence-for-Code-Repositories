from pathlib import Path
from app.parser.chunker import chunk_source
from scripts.verify_span_fidelity import run_span_fidelity


def test_span_fidelity_on_commander_repo():
    repo_path = Path("test_repos/commander")
    if not repo_path.exists():
        # Fallback to local files if repo not present
        repo_path = Path("demo_repo")
    passed, msg = run_span_fidelity(repo_path)
    assert passed, f"Span fidelity failed: {msg}"
    assert "100.0% exact match" in msg


def test_chunker_handles_complex_javascript_constructs():
    source = """
// Comment with { mismatched braces }
const multilineArrow = (a, b) => {
    const x = `template with ${"{ inner: 123 }"}`;
    return a + b + x.length;
};

const objWithMethods = {
    plainMethod(arg) {
        return arg * 2;
    },
    arrowProp: (val) => val + 1,
};

class BaseService {
    arrowField = (input) => {
        return input.toUpperCase();
    };

    constructor(name) {
        this.name = name;
    }

    async process() {
        return true;
    }

    *generator() {
        yield 1;
        yield 2;
    }
}

export default function exportedMain() {
    return 42;
}
"""
    chunks = chunk_source(source, "complex.js")
    raw_bytes = source.encode("utf-8")

    # Verify 100.0% exact span fidelity on every chunk
    for chunk in chunks:
        assert raw_bytes[chunk.start_byte:chunk.end_byte].decode("utf-8") == chunk.content

    types = {c.type for c in chunks}
    assert "FILE_SUMMARY" in types
    assert "FUNCTION" in types
    assert "METHOD" in types
    assert "CLASS" in types
    assert "CLASS_FIELD" in types

    names = {c.name for c in chunks}
    assert "multilineArrow" in names
    assert "plainMethod" in names
    assert "arrowProp" in names
    assert "BaseService" in names
    assert "arrowField" in names
    assert "constructor" in names
    assert "process" in names
    assert "generator" in names
    assert "exportedMain" in names
