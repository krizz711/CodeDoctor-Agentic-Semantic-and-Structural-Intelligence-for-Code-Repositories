from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maps file extensions to (Language object, queries dict)
# queries dict maps chunk_type -> S-expression query string
LANGUAGE_REGISTRY: dict[str, tuple[Any, dict[str, str]]] = {}

try:
    import tree_sitter
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_typescript

    JS_LANG = tree_sitter.Language(tree_sitter_javascript.language())
    TS_LANG = tree_sitter.Language(tree_sitter_typescript.language_typescript())
    PY_LANG = tree_sitter.Language(tree_sitter_python.language())

    # JavaScript/TypeScript queries
    JS_TS_QUERIES = {
        "FUNCTION": "(function_declaration name: (identifier) @name) @definition",
        "ARROW_FUNCTION": "(variable_declarator name: (identifier) @name value: (arrow_function) @definition) @full",
        "CLASS": "(class_declaration name: (identifier) @name) @definition",
        "METHOD": "(method_definition name: (property_identifier) @name) @definition",
        "EXPORTED_FUNCTION": "(export_statement declaration: (function_declaration name: (identifier) @name) @definition) @full",
    }

    # Python queries
    PY_QUERIES = {
        "FUNCTION": "(function_definition name: (identifier) @name) @definition",
        "CLASS": "(class_definition name: (identifier) @name) @definition",
        "DECORATED": "(decorated_definition) @definition",
    }

    for ext in [".js", ".jsx", ".mjs", ".cjs"]:
        LANGUAGE_REGISTRY[ext] = (JS_LANG, JS_TS_QUERIES)

    for ext in [".ts", ".tsx"]:
        LANGUAGE_REGISTRY[ext] = (TS_LANG, JS_TS_QUERIES)

    for ext in [".py"]:
        LANGUAGE_REGISTRY[ext] = (PY_LANG, PY_QUERIES)

    logger.debug("Tree-sitter languages loaded: JS, TS, Python")

except ImportError as e:
    logger.warning("Failed to load tree-sitter languages: %s. Falling back to regex parser.", e)
