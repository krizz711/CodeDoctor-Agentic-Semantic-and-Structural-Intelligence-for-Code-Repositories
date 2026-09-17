from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


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


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for idx, char in enumerate(text):
        if char == "\n":
            offsets.append(idx + 1)
    return offsets


def _locate_line(text: str, byte_offset: int) -> int:
    offsets = _line_offsets(text)
    line = 1
    for idx, offset in enumerate(offsets):
        if offset > byte_offset:
            return max(1, idx)
    return max(1, len(offsets))


def _extract_function_signature(match: re.Match[str]) -> str:
    snippet = match.group(0)
    if "=" in snippet:
        return snippet.strip().split("=")[0].strip()[:200]
    return snippet.strip()[:200]


def chunk_source(source: str, file_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    byte_index = 0
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
        byte_index = max(byte_index, end)

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
