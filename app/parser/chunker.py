from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    file_path: str
    type: str
    name: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent: str | None = None
    content: str = ""
    signature: str = ""

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.start_byte}:{self.end_byte}"


CONTROL_FLOW_KEYWORDS = {
    "if",
    "else",
    "while",
    "for",
    "switch",
    "catch",
    "do",
    "return",
    "throw",
    "new",
    "typeof",
    "instanceof",
    "void",
    "delete",
    "in",
    "of",
    "try",
    "finally",
}


def _extract_function_signature(match: re.Match[str]) -> str:
    snippet = match.group(0)
    if "=" in snippet:
        return snippet.strip().split("=")[0].strip()[:200]
    return snippet.strip()[:200]


def _fallback_chunk_source(source: str, file_path: str) -> list[Chunk]:
    """Regex-based fallback chunker for when tree-sitter is unavailable."""
    chunks: list[Chunk] = []
    pattern = re.compile(
        r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{",
        re.MULTILINE,
    )
    class_pattern = re.compile(
        r"(?:export\s+)?class\s+([A-Za-z0-9_]+)\s*\{",
        re.MULTILINE,
    )

    for match in pattern.finditer(source):
        name = match.group(1)
        start = match.start()
        brace_start = source.find("{", match.start())
        depth = 0
        end = None
        for i in range(brace_start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        content = source[start:end]
        line_start = source.count("\n", 0, start) + 1
        line_end = source.count("\n", 0, end) + 1
        chunks.append(
            Chunk(
                file_path=file_path,
                type="FUNCTION",
                name=name,
                start_byte=start,
                end_byte=end,
                start_line=line_start,
                end_line=line_end,
                content=content,
                signature=_extract_function_signature(match),
            )
        )

    for match in class_pattern.finditer(source):
        name = match.group(1)
        start = match.start()
        brace_start = source.find("{", match.start())
        depth = 0
        end = None
        for i in range(brace_start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        if any(chunk.name == name and chunk.start_byte == start for chunk in chunks):
            continue
        content = source[start:end]
        line_start = source.count("\n", 0, start) + 1
        line_end = source.count("\n", 0, end) + 1
        chunks.append(
            Chunk(
                file_path=file_path,
                type="CLASS",
                name=name,
                start_byte=start,
                end_byte=end,
                start_line=line_start,
                end_line=line_end,
                content=content,
                signature=f"class {name}",
            )
        )

    for method_match in re.finditer(r"(?:async\s+)?([A-Za-z0-9_]+)\s*\([^)]*\)\s*\{", source, re.MULTILINE):
        name = method_match.group(1)
        if name in CONTROL_FLOW_KEYWORDS:
            continue
        if name in {chunk.name for chunk in chunks if chunk.type == "FUNCTION"}:
            continue
        start = method_match.start()
        brace_start = source.find("{", start)
        depth = 0
        end = None
        for i in range(brace_start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue
        if any(chunk.start_byte == start for chunk in chunks):
            continue
        content = source[start:end]
        line_start = source.count("\n", 0, start) + 1
        line_end = source.count("\n", 0, end) + 1
        chunks.append(
            Chunk(
                file_path=file_path,
                type="METHOD",
                name=name,
                start_byte=start,
                end_byte=end,
                start_line=line_start,
                end_line=line_end,
                content=content,
                signature=f"{name}()",
            )
        )

    if not chunks:
        chunks.append(
            Chunk(
                file_path=file_path,
                type="FILE_SUMMARY",
                name="file",
                start_byte=0,
                end_byte=len(source),
                start_line=1,
                end_line=max(1, source.count("\n") + 1),
                content=source,
                signature="file",
            )
        )

    return sorted(chunks, key=lambda chunk: (chunk.start_byte, chunk.end_byte))


def _treesitter_chunk_source(source: str, file_path: str) -> list[Chunk]:
    """Tree-sitter AST-based chunker for precise code extraction."""
    import tree_sitter

    from app.parser.languages import LANGUAGE_REGISTRY

    _, ext = os.path.splitext(file_path)
    if ext not in LANGUAGE_REGISTRY:
        return []

    lang, queries = LANGUAGE_REGISTRY[ext]
    parser = tree_sitter.Parser(lang)
    tree = parser.parse(bytes(source, "utf8"))
    root_node = tree.root_node

    chunks: list[Chunk] = []
    seen_ranges: set[tuple[int, int]] = set()
    seen_names_at_line: set[tuple[str, int]] = set()  # (name, start_line) for dedup

    def _find_parent_class(node) -> str | None:
        """Walk up the tree to find an enclosing class name."""
        curr = node.parent
        while curr:
            if curr.type in ("class_declaration", "class_definition"):
                for child in curr.children:
                    if child.type == "identifier":
                        return source[child.start_byte : child.end_byte]
            curr = curr.parent
        return None

    for chunk_type, query_str in queries.items():
        query = tree_sitter.Query(lang, query_str)
        cursor = tree_sitter.QueryCursor(query)
        matches = list(cursor.matches(root_node))

        for _pattern_idx, capture_dict in matches:
            # capture_dict is {capture_name: [Node, ...]}
            target_node = None
            name_str = ""

            # Get the target node (the full definition)
            for key in ("full", "definition"):
                nodes = capture_dict.get(key, [])
                if nodes:
                    target_node = nodes[0]
                    break

            if target_node is None:
                continue

            # Get the name
            name_nodes = capture_dict.get("name", [])
            if name_nodes:
                n = name_nodes[0]
                name_str = source[n.start_byte : n.end_byte]
            elif chunk_type == "DECORATED":
                # For decorated definitions, dig into children for the function name
                for child in target_node.children:
                    if child.type in ("function_definition", "class_definition"):
                        for gc in child.children:
                            if gc.type == "identifier":
                                name_str = source[gc.start_byte : gc.end_byte]
                                break
                        break

            if not name_str:
                name_str = "unknown"

            start_line = target_node.start_point[0] + 1
            name_key = (name_str, start_line)
            if name_key in seen_names_at_line:
                continue
            seen_names_at_line.add(name_key)

            node_range = (target_node.start_byte, target_node.end_byte)
            if node_range in seen_ranges:
                continue
            seen_ranges.add(node_range)

            start_byte = target_node.start_byte
            end_byte = target_node.end_byte
            end_line = target_node.end_point[0] + 1
            content = source[start_byte:end_byte]
            sig_line = content.split("\n")[0][:200]

            # Determine chunk type
            final_type = "FUNCTION"
            if chunk_type == "CLASS":
                final_type = "CLASS"
            elif chunk_type == "METHOD":
                final_type = "METHOD"
            elif chunk_type == "DECORATED":
                # Check if it's a decorated class or function
                for child in target_node.children:
                    if child.type == "class_definition":
                        final_type = "CLASS"
                        break
                else:
                    parent_class = _find_parent_class(target_node)
                    final_type = "METHOD" if parent_class else "FUNCTION"
            else:
                parent_class = _find_parent_class(target_node)
                if parent_class:
                    final_type = "METHOD"

            parent_name = _find_parent_class(target_node) if final_type != "CLASS" else None

            chunks.append(
                Chunk(
                    file_path=file_path,
                    type=final_type,
                    name=name_str,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    start_line=start_line,
                    end_line=end_line,
                    parent=parent_name,
                    content=content,
                    signature=sig_line,
                )
            )

    return chunks


def chunk_source(source: str, file_path: str) -> list[Chunk]:
    """Parse source code into semantically meaningful chunks.

    Uses tree-sitter AST parsing when available, with a regex fallback.
    Returns a sorted list of Chunk objects with exact byte and line ranges.
    """
    # Try tree-sitter first
    try:
        from app.parser.languages import LANGUAGE_REGISTRY

        if LANGUAGE_REGISTRY:
            chunks = _treesitter_chunk_source(source, file_path)
            if chunks:
                return sorted(chunks, key=lambda c: (c.start_byte, c.end_byte))
    except Exception as e:
        logger.debug("Tree-sitter chunking failed, using fallback: %s", e)

    # Fall back to regex
    chunks = _fallback_chunk_source(source, file_path)
    if chunks:
        return chunks

    # Ultimate fallback: whole file as one chunk
    return [
        Chunk(
            file_path=file_path,
            type="FILE_SUMMARY",
            name="file",
            start_byte=0,
            end_byte=len(source),
            start_line=1,
            end_line=max(1, source.count("\n") + 1),
            content=source,
            signature="file",
        )
    ]
