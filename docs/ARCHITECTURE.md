# CodeDoctor Architecture Record

## 1. Ownership Boundary Decision (Phase 1, Task 1)

As specified in the build plan and `changes.md`, the system requires **one single shared AST chunker** to eliminate duplicate and divergent parsing logic.

### Decision: Python Tree-Sitter Core with Standardized Interop

We adopt **Option (b)**: Python `tree-sitter` (`tree_sitter_javascript`, `tree_sitter_python`) serves as the single authoritative AST parser and chunker for the system.

### Rationale:
1. **Toolchain & Runtime Reliability**: The Python runtime has native, battle-tested `tree-sitter` (0.25.2), `tree-sitter-javascript` (0.25.0), and `tree-sitter-python` (0.25.0) pre-installed and functional across both Windows and Linux container environments.
2. **Deterministic Span Arithmetic**: Tree-sitter node byte ranges (`node.start_byte`, `node.end_byte`) and point locations (`node.start_point`, `node.end_point`) are exact and natively provide 100.0% span fidelity when slicing raw source bytes.
3. **Multi-Grammar Support**: Python tree-sitter effortlessly handles JavaScript (`.js`, `.jsx`, `.mjs`, `.cjs`), TypeScript (`.ts`, `.tsx`), and Python (`.py`) under identical chunking semantics.
4. **C++ Interoperability**: The C++ product target retains its indexing, SQLite, and symbol search capabilities by importing standard chunk definitions (via TSV/JSON or CLI bridge) produced by the unified tree-sitter core, guaranteeing zero divergent behavior. The obsolete regex-and-brace-counting parser in C++ is decommissioned.

## 2. Chunk Types and Grammar Mapping

The unified chunker extracts the following AST node types:
- **`FUNCTION`**: Function declarations, generator functions, async functions, exported functions.
- **`ARROW_FUNCTION` / `FUNCTION`**: Arrow functions and function expressions assigned to variables/constants (`const fn = () => ...`).
- **`METHOD`**: Class methods, constructor methods, generator methods, async methods, and object literal methods (`{ method() {} }`).
- **`CLASS`**: Class declarations and exported classes.
- **`CLASS_FIELD`**: Class properties initialized with functions or arrow functions.
- **`MODULE_CODE`**: Significant top-level module statements (e.g. initialization logic).
- **`FILE_SUMMARY`**: Generated **for every file, always**, capturing module-level imports, exports, and high-level structure.

### Oversized Function Policy
Functions exceeding statement/line thresholds (>100 lines or >2000 characters) are partitioned at top-level statement boundaries within the function body. Each chunk retains:
- The parent identifier (`parent = function_name`)
- The original signature repeated at the beginning of the chunk representation
- Exact byte offsets corresponding to the specific slice in the source file.

## 3. Span Fidelity Invariant
For every chunk produced by the chunker:
$$\text{source}[\text{start\_byte}:\text{end\_byte}] == \text{chunk.content}$$
This invariant must hold across 100.0% of indexed chunks.

## 4. Configuration Hierarchy & Production Model Resolution (Remediation Task 1 & 5)

To guarantee reproducibility and insulation from active tuning:
1. **Config Hierarchy:** `config/frozen.yaml` takes precedence over `config/retrieval.yaml`. If `config/frozen.yaml` is present, all evaluation and production retrieval services lock into the frozen specification.
2. **Model Identifier:** `CodeEncoder` and `scripts/evaluate_mteb.py` read `encoder.production_model` as the single authoritative model identifier (pointing to `codefuse-ai/F2LLM-v2-0.6B`). The decorative `model_name` key was removed from the active configuration files to eliminate divergence between the screening harness and the runtime service.
3. **Instruction Prefixes:** Prompt prefixes are pinned per mode (`query` vs `document`) and automatically applied by `CodeDoctorMTEBEncoder` using MTEB 2.x `PromptType` dispatch.

