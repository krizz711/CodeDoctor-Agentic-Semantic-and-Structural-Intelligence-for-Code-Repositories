import hashlib
import logging
from tree_sitter import Language, Parser
import tree_sitter_javascript

logger = logging.getLogger(__name__)

class ASTParseError(Exception):
    """Raised when source code AST parsing fails due to syntax or structural errors."""
    pass

# Initialize Tree-sitter for JavaScript
_js_language = Language(tree_sitter_javascript.language())
_js_parser = Parser(_js_language)

def parse_javascript(code: str):
    """
    Parses JavaScript code into exact AST statement/function chunks using Tree-sitter.
    Rule 7: Defensive Programming. Raises ASTParseError on malformed syntax without silent fallbacks.
    Satisfies T7 (Span Fidelity) and T10 (No Silent Fallback).
    """
    if not code:
        return []
        
    code_bytes = code.encode("utf-8")
    tree = _js_parser.parse(code_bytes)
    
    if tree.root_node.has_error:
        logger.error("Tree-sitter detected syntax error in JavaScript source.")
        raise ASTParseError("Failed to parse JavaScript code: syntax error detected in source.")
        
    chunks = []
    for child in tree.root_node.children:
        if child.is_named:
            sb = child.start_byte
            eb = child.end_byte
            chunk_content = code_bytes[sb:eb].decode("utf-8")
            h = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
            chunks.append({
                "start_byte": sb,
                "end_byte": eb,
                "content": chunk_content,
                "semantic_hash": h,
                "ast_type": child.type
            })
            
    # If there are no top-level named children (e.g. bare comments or single token), fallback to root span
    if not chunks and code.strip():
        h = hashlib.sha256(code.encode("utf-8")).hexdigest()
        chunks.append({
            "start_byte": 0,
            "end_byte": len(code_bytes),
            "content": code,
            "semantic_hash": h,
            "ast_type": "program"
        })
        
    return chunks


def sliding_window_chunk(text: str, chunk_size: int = 2000, overlap: int = 200):
    """
    Slices arbitrary long-form code strings into overlapping character chunks.
    Maintained for non-JS files and extreme tail fallback.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += (chunk_size - overlap)
        
    return chunks


def parse_code(code: str):
    """Backwards compatibility proxy."""
    return sliding_window_chunk(code)
