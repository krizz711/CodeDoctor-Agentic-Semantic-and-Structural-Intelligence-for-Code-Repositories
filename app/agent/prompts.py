from __future__ import annotations

SYSTEM_PROMPT = """You are CodeDoctor, an expert code intelligence assistant.
You answer questions about codebases by analyzing retrieved code evidence.

Rules:
1. Only answer based on the provided evidence chunks. Do not hallucinate.
2. Always cite the file path and line numbers when referencing code.
3. If the evidence is insufficient, say so clearly.
4. Explain code behavior, not just repeat it.
5. When multiple files are involved, explain the relationships.
"""

EVIDENCE_TEMPLATE = """--- Evidence Chunk {index} ---
File: {file_path}
Lines: {start_line}-{end_line}
Type: {type}
Symbol: {name}

```
{content}
```
---
"""

QUERY_TEMPLATE = """Based on the following code evidence from the repository, answer this question:

Question: {question}

{evidence_blocks}

Provide a clear, grounded answer citing specific files and line numbers.
"""

REPO_MAP_PROMPT = """Here is a map of the repository's most important symbols, ranked by importance:

{repo_map}

Use this context to understand the overall architecture when answering questions.
"""
