from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.graph.call_graph import CodeCallGraph


def generate_repo_map(
    graph: CodeCallGraph,
    symbols: dict[str, dict],  # symbol_name -> {file_path, type, signature, start_line}
    max_tokens: int = 4000,
) -> str:
    """Generate a token-budgeted repository map using PageRank.

    Inspired by Aider's repo-map strategy: uses PageRank to identify
    the most important symbols in the codebase and formats them into
    a concise summary that fits within an LLM's context window.
    """
    ranks = graph.pagerank() if graph.node_count > 0 else {}

    # Sort symbols by PageRank importance
    ranked_symbols = sorted(
        symbols.items(),
        key=lambda x: ranks.get(x[0], 0.0),
        reverse=True,
    )

    lines = ["# Repository Map (ranked by importance)\n"]
    current_file = None
    estimated_tokens = 10  # header overhead

    for name, info in ranked_symbols:
        file_path = info.get("file_path", "")
        sym_type = info.get("type", "UNKNOWN")
        signature = info.get("signature", name)
        line_num = info.get("start_line", 0)
        if file_path != current_file:
            file_line = f"\n## {file_path}\n"
            lines.append(file_line)
            current_file = file_path
            estimated_tokens += len(file_line.split()) + 2

        entry = f"  L{line_num}: {sym_type.lower()} {signature}"
        callers = graph.callers_of(name)
        callees = graph.callees_of(name)
        if callers or callees:
            refs = []
            if callers:
                refs.append(f"called by: {', '.join(callers[:3])}")
            if callees:
                refs.append(f"calls: {', '.join(callees[:3])}")
            entry += f"  # {'; '.join(refs)}"

        entry_tokens = len(entry.split()) + 1
        if estimated_tokens + entry_tokens > max_tokens:
            lines.append(f"\n# ... ({len(ranked_symbols) - len(lines)} more symbols truncated)")
            break

        lines.append(entry)
        estimated_tokens += entry_tokens

    return "\n".join(lines)
