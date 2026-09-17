from __future__ import annotations

from app.parser.chunker import chunk_source


def test_js_function_extraction():
    code = "function loginUser(email, password) { return verify(email); }"
    chunks = chunk_source(code, "test.js")
    funcs = [c for c in chunks if c.type == "FUNCTION"]
    assert len(funcs) >= 1
    assert funcs[0].name == "loginUser"


def test_js_class_and_methods():
    code = """class AuthService {
    constructor(db) { this.db = db; }
    async login(user, pass) { return this.db.find(user); }
    logout() { this.token = null; }
}"""
    chunks = chunk_source(code, "test.js")
    classes = [c for c in chunks if c.type == "CLASS"]
    methods = [c for c in chunks if c.type == "METHOD"]
    assert len(classes) >= 1
    assert classes[0].name == "AuthService"
    assert len(methods) >= 2  # constructor, login, logout
    for m in methods:
        assert m.parent == "AuthService"


def test_arrow_function_detection():
    code = "const helper = () => { return true; };"
    chunks = chunk_source(code, "test.js")
    # Should detect as a function (arrow function)
    assert any(c.name == "helper" for c in chunks)


def test_no_phantom_control_flow_methods():
    """Verify that if/while/for/catch are never captured as methods."""
    code = """function process(data) {
    if (data) { console.log('ok'); }
    while (running) { tick(); }
    for (let i = 0; i < 10; i++) { step(i); }
    try { run(); } catch (err) { handle(err); }
}"""
    chunks = chunk_source(code, "test.js")
    names = {c.name for c in chunks}
    assert "if" not in names
    assert "while" not in names
    assert "for" not in names
    assert "catch" not in names


def test_exported_function():
    code = "export function handleRequest(req, res) { return res.send(200); }"
    chunks = chunk_source(code, "test.js")
    # Should detect exactly one function, not duplicate
    funcs = [c for c in chunks if c.name == "handleRequest"]
    assert len(funcs) == 1


def test_python_function_extraction():
    code = '''def calculate_sum(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def add(self, x, y):
        return x + y
    def subtract(self, x, y):
        return x - y
'''
    chunks = chunk_source(code, "test.py")
    assert any(c.name == "calculate_sum" and c.type == "FUNCTION" for c in chunks)
    assert any(c.name == "Calculator" and c.type == "CLASS" for c in chunks)
    methods = [c for c in chunks if c.type == "METHOD"]
    assert len(methods) >= 2


def test_empty_file_returns_file_summary():
    chunks = chunk_source("", "test.js")
    assert len(chunks) == 1
    assert chunks[0].type == "FILE_SUMMARY"


def test_chunk_has_correct_line_numbers():
    code = """// header comment
function foo() {
    return 42;
}
"""
    chunks = chunk_source(code, "test.js")
    funcs = [c for c in chunks if c.type == "FUNCTION"]
    if funcs:
        assert funcs[0].start_line == 2
        assert funcs[0].end_line == 4


def test_unsupported_extension_returns_file_summary():
    chunks = chunk_source("some random code", "test.rs")
    assert len(chunks) == 1
    assert chunks[0].type == "FILE_SUMMARY"


def test_chunk_content_matches_source():
    code = 'function greet(name) { return "Hello " + name; }'
    chunks = chunk_source(code, "test.js")
    for c in chunks:
        if c.type != "FILE_SUMMARY":
            assert c.content == code[c.start_byte : c.end_byte]


def test_chunk_id_format():
    code = "function test() { return 1; }"
    chunks = chunk_source(code, "example.js")
    for c in chunks:
        assert c.chunk_id.startswith("example.js:")
