from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List
from tree_sitter import Node

from app.parser.chunker import _compute_line, _get_parser_and_lang


@dataclass
class EdgeRecord:
    source: str
    target: str
    relation: str  # 'CALLS', 'IMPORTS', 'REQUIRES', 'EXPORTS', 'REFERENCES_DEEPLINK'
    confidence: float
    status: str  # 'VERIFIED' or 'CANDIDATE'
    file_path: str
    line: int
    start_byte: int
    end_byte: int
    citing_snippet: str
    extractor: str = "tree-sitter"

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "confidence": self.confidence,
            "status": self.status,
            "file_path": self.file_path,
            "line": self.line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "citing_snippet": self.citing_snippet,
            "extractor": self.extractor,
        }


def _node_text(raw_bytes: bytes, node: Node) -> str:
    return raw_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


class StructuralExtractor:
    """
    Extracts high-fidelity structural relationships (calls, imports, requires, exports, deeplinks)
    with strict 'VERIFIED' vs 'CANDIDATE' honest labelling.
    """

    def __init__(self) -> None:
        pass

    def extract_file_edges(self, file_path: str, source: str) -> List[EdgeRecord]:
        raw_bytes = source.encode("utf-8")
        parser, lang = _get_parser_and_lang(file_path)
        tree = parser.parse(raw_bytes)
        root = tree.root_node

        edges: List[EdgeRecord] = []
        ext = Path(file_path).suffix.lower()

        if ext == ".py":
            self._extract_python_edges(raw_bytes, file_path, root, edges)
        else:
            self._extract_javascript_edges(raw_bytes, file_path, root, edges)

        return edges

    def _extract_javascript_edges(
        self, raw_bytes: bytes, file_path: str, root_node: Node, edges: List[EdgeRecord]
    ) -> None:
        def get_enclosing_function_name(node: Node) -> str:
            curr = node.parent
            while curr:
                if curr.type in ("function_declaration", "generator_function_declaration"):
                    n = curr.child_by_field_name("name")
                    return _node_text(raw_bytes, n) if n else "anonymous"
                elif curr.type == "method_definition":
                    n = curr.child_by_field_name("name")
                    return _node_text(raw_bytes, n) if n else "method"
                elif curr.type == "variable_declarator":
                    n = curr.child_by_field_name("name")
                    v = curr.child_by_field_name("value")
                    if n and v and v.type in ("arrow_function", "function_expression"):
                        return _node_text(raw_bytes, n)
                curr = curr.parent
            return f"module::{Path(file_path).stem}"

        def visit(node: Node):
            ntype = node.type

            # 1. Imports
            if ntype == "import_statement":
                source_node = node.child_by_field_name("source")
                import_src = _node_text(raw_bytes, source_node).strip("'\"") if source_node else ""
                # Named imports or default
                snippet = _node_text(raw_bytes, node)
                line = _compute_line(raw_bytes, node.start_byte)
                for child in node.children:
                    if child.type == "import_clause":
                        clause_text = _node_text(raw_bytes, child)
                        # Extract imported symbols
                        for sym in re.findall(r"[A-Za-z0-9_]+", clause_text):
                            if sym not in ("import", "from", "as", "type", "default"):
                                edges.append(
                                    EdgeRecord(
                                        source=file_path,
                                        target=sym,
                                        relation="IMPORTS",
                                        confidence=1.0,
                                        status="VERIFIED",
                                        file_path=file_path,
                                        line=line,
                                        start_byte=node.start_byte,
                                        end_byte=node.end_byte,
                                        citing_snippet=snippet,
                                    )
                                )

            # 2. CommonJS Requires
            elif ntype == "call_expression":
                func_node = node.child_by_field_name("function")
                args_node = node.child_by_field_name("arguments")
                caller_name = get_enclosing_function_name(node)
                line = _compute_line(raw_bytes, node.start_byte)
                snippet = _node_text(raw_bytes, node)

                if func_node and _node_text(raw_bytes, func_node) == "require":
                    # require('./module')
                    req_path = _node_text(raw_bytes, args_node).strip("() '\"") if args_node else ""
                    edges.append(
                        EdgeRecord(
                            source=file_path,
                            target=req_path,
                            relation="REQUIRES",
                            confidence=1.0,
                            status="VERIFIED",
                            file_path=file_path,
                            line=line,
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            citing_snippet=snippet,
                        )
                    )
                elif func_node:
                    # Direct function call or method call
                    if func_node.type == "identifier":
                        callee = _node_text(raw_bytes, func_node)
                        edges.append(
                            EdgeRecord(
                                source=caller_name,
                                target=callee,
                                relation="CALLS",
                                confidence=1.0,
                                status="VERIFIED",
                                file_path=file_path,
                                line=line,
                                start_byte=node.start_byte,
                                end_byte=node.end_byte,
                                citing_snippet=snippet,
                            )
                        )
                    elif func_node.type == "subscript_expression":
                        # Computed property call: dispatcher[methodName]()
                        idx_node = func_node.child_by_field_name("index")
                        callee = _node_text(raw_bytes, idx_node) if idx_node else "dynamic"
                        edges.append(
                            EdgeRecord(
                                source=caller_name,
                                target=callee,
                                relation="CALLS",
                                confidence=0.65,
                                status="CANDIDATE",
                                file_path=file_path,
                                line=line,
                                start_byte=node.start_byte,
                                end_byte=node.end_byte,
                                citing_snippet=snippet,
                            )
                        )
                    elif func_node.type == "member_expression":
                        # Object method call: obj.method()
                        prop_node = func_node.child_by_field_name("property")
                        if prop_node:
                            callee = _node_text(raw_bytes, prop_node)
                            edges.append(
                                EdgeRecord(
                                    source=caller_name,
                                    target=callee,
                                    relation="CALLS",
                                    confidence=0.95,
                                    status="VERIFIED",
                                    file_path=file_path,
                                    line=line,
                                    start_byte=node.start_byte,
                                    end_byte=node.end_byte,
                                    citing_snippet=snippet,
                                )
                            )

            # 3. Exports
            elif ntype == "export_statement":
                snippet = _node_text(raw_bytes, node)
                line = _compute_line(raw_bytes, node.start_byte)
                for child in node.children:
                    if child.type in ("function_declaration", "class_declaration"):
                        n = child.child_by_field_name("name")
                        if n:
                            exported_name = _node_text(raw_bytes, n)
                            edges.append(
                                EdgeRecord(
                                    source=exported_name,
                                    target=file_path,
                                    relation="EXPORTS",
                                    confidence=1.0,
                                    status="VERIFIED",
                                    file_path=file_path,
                                    line=line,
                                    start_byte=node.start_byte,
                                    end_byte=node.end_byte,
                                    citing_snippet=snippet,
                                )
                            )

            # 4. String literal references & Deeplinks (e.g. bluetooth-settings)
            elif ntype in ("string", "template_string"):
                text_val = _node_text(raw_bytes, node).strip("'\"`")
                if any(kw in text_val.lower() for kw in ("bluetooth", "settings", "deeplink", "action:")):
                    caller_name = get_enclosing_function_name(node)
                    line = _compute_line(raw_bytes, node.start_byte)
                    snippet = _node_text(raw_bytes, node)
                    edges.append(
                        EdgeRecord(
                            source=caller_name,
                            target=text_val,
                            relation="REFERENCES_DEEPLINK",
                            confidence=1.0,
                            status="VERIFIED",
                            file_path=file_path,
                            line=line,
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            citing_snippet=snippet,
                        )
                    )

            for child in node.children:
                visit(child)

        visit(root_node)

    def _extract_python_edges(
        self, raw_bytes: bytes, file_path: str, root_node: Node, edges: List[EdgeRecord]
    ) -> None:
        def get_enclosing_function(node: Node) -> str:
            curr = node.parent
            while curr:
                if curr.type in ("function_definition", "async_function_definition"):
                    n = curr.child_by_field_name("name")
                    return _node_text(raw_bytes, n) if n else "function"
                curr = curr.parent
            return f"module::{Path(file_path).stem}"

        def visit(node: Node):
            ntype = node.type
            if ntype in ("import_statement", "import_from_statement"):
                snippet = _node_text(raw_bytes, node)
                line = _compute_line(raw_bytes, node.start_byte)
                for child in node.children:
                    if child.type in ("dotted_name", "aliased_import", "identifier"):
                        sym = _node_text(raw_bytes, child)
                        edges.append(
                            EdgeRecord(
                                source=file_path,
                                target=sym,
                                relation="IMPORTS",
                                confidence=1.0,
                                status="VERIFIED",
                                file_path=file_path,
                                line=line,
                                start_byte=node.start_byte,
                                end_byte=node.end_byte,
                                citing_snippet=snippet,
                            )
                        )
            elif ntype == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    caller = get_enclosing_function(node)
                    line = _compute_line(raw_bytes, node.start_byte)
                    snippet = _node_text(raw_bytes, node)
                    callee = _node_text(raw_bytes, func_node)
                    # If attribute call obj.method()
                    is_candidate = func_node.type == "attribute"
                    status = "CANDIDATE" if is_candidate else "VERIFIED"
                    conf = 0.75 if is_candidate else 1.0
                    edges.append(
                        EdgeRecord(
                            source=caller,
                            target=callee.split(".")[-1],
                            relation="CALLS",
                            confidence=conf,
                            status=status,
                            file_path=file_path,
                            line=line,
                            start_byte=node.start_byte,
                            end_byte=node.end_byte,
                            citing_snippet=snippet,
                        )
                    )
            for child in node.children:
                visit(child)

        visit(root_node)
