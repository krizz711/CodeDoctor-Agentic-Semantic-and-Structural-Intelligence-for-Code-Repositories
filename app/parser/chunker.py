from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_python

try:
    import tree_sitter_typescript
    _HAS_TS = True
except ImportError:
    _HAS_TS = False

# Initialize languages and parsers
JS_LANG = Language(tree_sitter_javascript.language())
PY_LANG = Language(tree_sitter_python.language())
if _HAS_TS:
    TS_LANG = Language(tree_sitter_typescript.language_typescript())
    TSX_LANG = Language(tree_sitter_typescript.language_tsx())
else:
    TS_LANG = JS_LANG
    TSX_LANG = JS_LANG


@dataclass
class Chunk:
    file_path: str
    type: str
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent: Optional[str] = None
    content: str = ""
    signature: str = ""

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.type}:{self.name}:{self.start_byte}:{self.end_byte}"


def _get_parser_and_lang(file_path: str) -> tuple[Parser, Language]:
    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        return Parser(PY_LANG), PY_LANG
    elif ext == ".ts":
        return Parser(TS_LANG), TS_LANG
    elif ext == ".tsx":
        return Parser(TSX_LANG), TSX_LANG
    else:
        return Parser(JS_LANG), JS_LANG


def _compute_line(raw_bytes: bytes, byte_offset: int) -> int:
    return raw_bytes[:byte_offset].count(b"\n") + 1


def _node_text(raw_bytes: bytes, node: Node) -> str:
    return raw_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _extract_js_signature(raw_bytes: bytes, node: Node) -> str:
    """Extract declaration header up to the opening brace or arrow."""
    text = _node_text(raw_bytes, node).strip()
    brace_idx = text.find("{")
    if brace_idx != -1:
        sig = text[:brace_idx].strip()
        if sig:
            return sig
    arrow_idx = text.find("=>")
    if arrow_idx != -1:
        sig = text[:arrow_idx + 2].strip()
        if sig:
            return sig
    newline_idx = text.find("\n")
    if newline_idx != -1:
        return text[:newline_idx].strip()
    return text[:200]


def _extract_py_signature(raw_bytes: bytes, node: Node) -> str:
    """Extract def/class header line."""
    text = _node_text(raw_bytes, node).strip()
    colon_idx = text.find(":")
    if colon_idx != -1:
        return text[:colon_idx + 1].strip()
    newline_idx = text.find("\n")
    if newline_idx != -1:
        return text[:newline_idx].strip()
    return text[:200]


def _split_oversized_function(
    raw_bytes: bytes,
    file_path: str,
    node: Node,
    name: str,
    parent: Optional[str],
    signature: str,
    chunk_type: str = "FUNCTION",
    max_lines: int = 80,
    max_bytes: int = 2500,
) -> list[Chunk]:
    """Split an oversized function at top-level statement boundaries."""
    node_bytes_len = node.end_byte - node.start_byte
    line_count = _compute_line(raw_bytes, node.end_byte) - _compute_line(raw_bytes, node.start_byte) + 1

    if line_count <= max_lines and node_bytes_len <= max_bytes:
        return []

    # Find the body block
    body_node = None
    for child in node.children:
        if child.type in ("statement_block", "block", "class_body"):
            body_node = child
            break

    if not body_node or not body_node.children:
        return []

    # Collect meaningful statement children (skip braces/colons)
    statements = [
        child for child in body_node.children
        if child.type not in ("{", "}", ":", ";") and (child.end_byte > child.start_byte)
    ]

    if len(statements) < 2:
        return []

    pieces: list[Chunk] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    accumulated_lines = 0

    def flush_piece():
        nonlocal current_start, current_end, accumulated_lines
        if current_start is not None and current_end is not None and current_end > current_start:
            content = raw_bytes[current_start:current_end].decode("utf-8", errors="replace")
            s_line = _compute_line(raw_bytes, current_start)
            e_line = _compute_line(raw_bytes, current_end)
            part_idx = len(pieces) + 1
            pieces.append(
                Chunk(
                    file_path=file_path,
                    type=chunk_type,
                    name=f"{name}_part{part_idx}",
                    start_byte=current_start,
                    end_byte=current_end,
                    start_line=s_line,
                    end_line=e_line,
                    parent=name,
                    content=content,
                    signature=signature,
                )
            )
        current_start = None
        current_end = None
        accumulated_lines = 0

    for stmt in statements:
        s_line = _compute_line(raw_bytes, stmt.start_byte)
        e_line = _compute_line(raw_bytes, stmt.end_byte)
        stmt_lines = e_line - s_line + 1

        if current_start is None:
            current_start = stmt.start_byte
            current_end = stmt.end_byte
            accumulated_lines = stmt_lines
        else:
            if accumulated_lines + stmt_lines > 40 or (stmt.end_byte - current_start) > 1500:
                flush_piece()
                current_start = stmt.start_byte
                current_end = stmt.end_byte
                accumulated_lines = stmt_lines
            else:
                current_end = stmt.end_byte
                accumulated_lines += stmt_lines

    flush_piece()
    return pieces


def _chunk_javascript(raw_bytes: bytes, file_path: str, root_node: Node) -> List[Chunk]:
    chunks: List[Chunk] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_chunk(
        node: Node,
        chunk_type: str,
        name: str,
        parent: Optional[str] = None,
        signature: str = "",
        allow_split: bool = True,
    ):
        span = (node.start_byte, node.end_byte)
        if span in seen_spans or node.end_byte <= node.start_byte:
            return
        seen_spans.add(span)

        content = raw_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = _compute_line(raw_bytes, node.start_byte)
        end_line = _compute_line(raw_bytes, node.end_byte)

        if not signature:
            signature = _extract_js_signature(raw_bytes, node)

        # Check if oversized
        if allow_split:
            parts = _split_oversized_function(
                raw_bytes, file_path, node, name, parent, signature, chunk_type=chunk_type
            )
            if parts:
                chunks.extend(parts)
                return

        chunks.append(
            Chunk(
                file_path=file_path,
                type=chunk_type,
                name=name,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=start_line,
                end_line=end_line,
                parent=parent,
                content=content,
                signature=signature,
            )
        )

    def visit(node: Node, current_class: Optional[str] = None, parent_name: Optional[str] = None):
        ntype = node.type

        if ntype == "export_statement":
            # Unwrap exported declaration if present, or let children be visited
            for child in node.children:
                if child.type in (
                    "function_declaration", "generator_function_declaration",
                    "class_declaration", "lexical_declaration", "variable_declaration"
                ):
                    visit(child, current_class=current_class, parent_name=parent_name)
                    return
            # If export default expression
            for child in node.children:
                if child.type in ("arrow_function", "function_expression"):
                    add_chunk(node, "FUNCTION", "default_export", parent=parent_name)
                    return

        elif ntype in ("function_declaration", "generator_function_declaration"):
            name_node = node.child_by_field_name("name")
            name = _node_text(raw_bytes, name_node) if name_node else "anonymous"
            sig = _extract_js_signature(raw_bytes, node)
            add_chunk(node, "FUNCTION", name, parent=current_class or parent_name, signature=sig)
            return

        elif ntype in ("lexical_declaration", "variable_declaration"):
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    if name_node and value_node and value_node.type in ("arrow_function", "function_expression"):
                        name = _node_text(raw_bytes, name_node)
                        sig = _extract_js_signature(raw_bytes, node)
                        add_chunk(node, "FUNCTION", name, parent=current_class or parent_name, signature=sig)
                        return
                    elif name_node and value_node and value_node.type == "object":
                        # Object with possible methods
                        obj_name = _node_text(raw_bytes, name_node)
                        for prop in value_node.children:
                            if prop.type == "method_definition":
                                m_name_node = prop.child_by_field_name("name")
                                m_name = _node_text(raw_bytes, m_name_node) if m_name_node else "method"
                                sig = _extract_js_signature(raw_bytes, prop)
                                add_chunk(prop, "METHOD", m_name, parent=obj_name, signature=sig)
                            elif prop.type == "pair":
                                key_node = prop.child_by_field_name("key")
                                val_node = prop.child_by_field_name("value")
                                if key_node and val_node and val_node.type in ("arrow_function", "function_expression"):
                                    m_name = _node_text(raw_bytes, key_node)
                                    sig = _extract_js_signature(raw_bytes, prop)
                                    add_chunk(prop, "METHOD", m_name, parent=obj_name, signature=sig)
                        return

        elif ntype == "class_declaration":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(raw_bytes, name_node) if name_node else "AnonymousClass"
            sig = f"class {class_name}"
            add_chunk(node, "CLASS", class_name, parent=parent_name, signature=sig, allow_split=False)
            # Visit methods and fields inside class body
            body_node = node.child_by_field_name("body")
            if body_node:
                for member in body_node.children:
                    if member.type == "method_definition":
                        m_name_node = member.child_by_field_name("name")
                        m_name = _node_text(raw_bytes, m_name_node) if m_name_node else "method"
                        sig = _extract_js_signature(raw_bytes, member)
                        add_chunk(member, "METHOD", m_name, parent=class_name, signature=sig)
                    elif member.type == "field_definition":
                        prop_name_node = member.child_by_field_name("property")
                        val_node = member.child_by_field_name("value")
                        if prop_name_node and val_node and val_node.type in ("arrow_function", "function_expression"):
                            f_name = _node_text(raw_bytes, prop_name_node)
                            sig = _extract_js_signature(raw_bytes, member)
                            add_chunk(member, "CLASS_FIELD", f_name, parent=class_name, signature=sig)
            return

        for child in node.children:
            visit(child, current_class=current_class, parent_name=parent_name)

    visit(root_node)
    return chunks


def _chunk_python(raw_bytes: bytes, file_path: str, root_node: Node) -> List[Chunk]:
    chunks: List[Chunk] = []
    seen_spans: set[tuple[int, int]] = set()

    def add_chunk(
        node: Node,
        chunk_type: str,
        name: str,
        parent: Optional[str] = None,
        signature: str = "",
        allow_split: bool = True,
    ):
        span = (node.start_byte, node.end_byte)
        if span in seen_spans or node.end_byte <= node.start_byte:
            return
        seen_spans.add(span)

        content = raw_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        start_line = _compute_line(raw_bytes, node.start_byte)
        end_line = _compute_line(raw_bytes, node.end_byte)

        if not signature:
            signature = _extract_py_signature(raw_bytes, node)

        if allow_split:
            parts = _split_oversized_function(
                raw_bytes, file_path, node, name, parent, signature, chunk_type=chunk_type
            )
            if parts:
                chunks.extend(parts)
                return

        chunks.append(
            Chunk(
                file_path=file_path,
                type=chunk_type,
                name=name,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=start_line,
                end_line=end_line,
                parent=parent,
                content=content,
                signature=signature,
            )
        )

    def visit(node: Node, current_class: Optional[str] = None):
        ntype = node.type

        if ntype == "function_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(raw_bytes, name_node) if name_node else "function"
            sig = _extract_py_signature(raw_bytes, node)
            c_type = "METHOD" if current_class else "FUNCTION"
            add_chunk(node, c_type, name, parent=current_class, signature=sig)
            # Do not recurse into nested defs as separate chunks unless desired
            return

        elif ntype == "class_definition":
            name_node = node.child_by_field_name("name")
            class_name = _node_text(raw_bytes, name_node) if name_node else "Class"
            sig = f"class {class_name}"
            add_chunk(node, "CLASS", class_name, parent=None, signature=sig, allow_split=False)
            body_node = node.child_by_field_name("body")
            if body_node:
                for child in body_node.children:
                    visit(child, current_class=class_name)
            return

        for child in node.children:
            visit(child, current_class=current_class)

    visit(root_node)
    return chunks


def chunk_source(source: str, file_path: str) -> List[Chunk]:
    """
    Unified AST chunker using Tree-Sitter for JavaScript/TypeScript and Python.
    Guarantees 100.0% span fidelity: source_bytes[chunk.start_byte:chunk.end_byte] == chunk.content.encode('utf-8').
    Always produces a FILE_SUMMARY chunk for every file.
    """
    raw_bytes = source.encode("utf-8")
    parser, lang = _get_parser_and_lang(file_path)
    tree = parser.parse(raw_bytes)
    root = tree.root_node

    ext = Path(file_path).suffix.lower()
    if ext == ".py":
        ast_chunks = _chunk_python(raw_bytes, file_path, root)
    else:
        ast_chunks = _chunk_javascript(raw_bytes, file_path, root)

    # Always generate a FILE_SUMMARY chunk for every file
    file_summary_chunk = Chunk(
        file_path=file_path,
        type="FILE_SUMMARY",
        name="file_summary",
        start_byte=0,
        end_byte=len(raw_bytes),
        start_line=1,
        end_line=max(1, _compute_line(raw_bytes, len(raw_bytes))),
        parent=None,
        content=source,
        signature="file_summary",
    )

    all_chunks = [file_summary_chunk] + ast_chunks
    return sorted(all_chunks, key=lambda c: (c.start_byte, c.end_byte))
