import pytest
from app.chunker import parse_javascript, ASTParseError


def test_t7_span_fidelity():
    """
    T7: Span Fidelity. 
    The parser must extract exact byte offsets. We verify that extracting the slice
    from the original string using the returned offsets perfectly matches the `content` field.
    """
    raw_code = "function valid() { return true; }"
    
    chunks = parse_javascript(raw_code)
    assert len(chunks) == 1
    
    chunk = chunks[0]
    extracted_slice = raw_code[chunk["start_byte"]:chunk["end_byte"]]
    
    assert extracted_slice == chunk["content"], "T7 FAILED: Extracted string does not match byte offsets."
    assert "{" in chunk["content"] and "}" in chunk["content"], "T7 FAILED: Missing block braces."

def test_t10_no_silent_fallback():
    """
    T10: No Silent Fallback.
    If the JS code is malformed, the parser MUST raise an ASTParseError.
    It must not silently fallback to regex parsing, ensuring strict structural integrity (Rule 1 & 7).
    """
    malformed_code = "function MALFORMED() { const x = ; return } } }"
    
    with pytest.raises(ASTParseError, match="Failed to parse"):
        parse_javascript(malformed_code)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
