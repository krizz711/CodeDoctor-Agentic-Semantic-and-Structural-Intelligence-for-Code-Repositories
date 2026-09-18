from pathlib import Path

from app.parser.chunker import Chunk, chunk_source


SOURCE = """
export function loginUser(email, password) {
    if (!email) {
        throw new Error('missing email');
    }
    return { email, password };
}

class AuthService {
    async getUserData() {
        return { ok: true };
    }
}
"""


def test_chunk_source_returns_exact_spans():
    chunks = chunk_source(SOURCE, "src/auth.js")
    assert chunks

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert chunk.content == SOURCE[chunk.start_byte:chunk.end_byte]


def test_chunk_source_extracts_function_and_class_names():
    chunks = chunk_source(SOURCE, "src/auth.js")
    names = {chunk.name for chunk in chunks}
    assert "loginUser" in names
    assert "AuthService" in names
    assert "getUserData" in names
