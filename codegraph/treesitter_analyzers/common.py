"""
treesitter_analyzers.common
----------------------------
Shared plumbing for every language analyzer: parser loading, generic
tree-walking, source-text slicing, and the same qualified_name()
convention your Python analyzers use (Flow_analyser._common), so
function ids look identical across languages: "ClassName.method" when
nested in a class, or just "method" at module scope.

Requires:  pip install "tree_sitter==0.21.3" tree_sitter_languages
(tree_sitter_languages ships prebuilt grammars for all 6 target
languages, so nothing needs to be compiled locally -- but its ABI only
matches tree_sitter<=0.21.x, NOT the newer 0.22+ API. Pin the version.)
"""

from __future__ import annotations

import warnings
from functools import lru_cache

# tree_sitter_languages triggers a harmless FutureWarning on import
# under tree_sitter==0.21.3 -- silence it so it doesn't spam every run.
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

from tree_sitter_languages import get_parser  # noqa: E402

# ---------------------------------------------------------------------------
# Language registry -- extension -> (tree-sitter grammar name, display name)
# The display name matches feature_extraction.py's fixed LANGUAGES vocab
# exactly, so node features built downstream don't need any remapping.
# ---------------------------------------------------------------------------
EXTENSION_LANGUAGE_MAP = {
    ".js": ("javascript", "JavaScript"),
    ".jsx": ("javascript", "JavaScript"),
    ".mjs": ("javascript", "JavaScript"),
    ".cjs": ("javascript", "JavaScript"),
    ".ts": ("typescript", "TypeScript"),
    ".tsx": ("tsx", "TypeScript"),
    ".go": ("go", "Go"),
    ".java": ("java", "Java"),
    ".c": ("c", "C"),
    ".h": ("c", "C"),
    ".cpp": ("cpp", "C++"),
    ".cc": ("cpp", "C++"),
    ".cxx": ("cpp", "C++"),
    ".hpp": ("cpp", "C++"),
    ".hh": ("cpp", "C++"),
}


@lru_cache(maxsize=None)
def get_cached_parser(grammar_name: str):
    """tree_sitter_languages.get_parser() reloads the .so on every call
    without caching -- cache it ourselves since we call this once per
    file, potentially thousands of times per repo."""
    return get_parser(grammar_name)


def parse_source(source_bytes: bytes, grammar_name: str):
    parser = get_cached_parser(grammar_name)
    return parser.parse(source_bytes)


def node_text(node, source_bytes: bytes) -> str:
    if node is None:
        return ""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def iter_nodes(node):
    """Depth-first PRE-ORDER walk of every descendant, root included --
    the tree-sitter equivalent of ast.walk(). Children must be pushed in
    REVERSED order so the stack-pop sequence matches left-to-right
    source order; without the reversal this silently returns nodes in
    a scrambled order, which breaks "first identifier" lookups (e.g.
    picking a parameter name instead of the function name in C/C++
    declarator chains)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def qualified_name(class_stack: list[str], name: str) -> str:
    """Same convention as Flow_analyser._common.qualified_name: dotted
    path through enclosing classes, e.g. ["Server"] + "start" ->
    "Server.start". Matches so downstream func_id()/class_id() joins in
    graph.py behave identically regardless of source language."""
    return ".".join(class_stack + [name]) if class_stack else name


def first_identifier_text(node, source_bytes: bytes, identifier_types=(
    "identifier", "field_identifier", "property_identifier",
    "type_identifier", "name",
)) -> str | None:
    """Finds the first identifier-like leaf under `node`. Used for the
    C/C++ declarator chains where the actual function/variable name is
    buried under nested pointer_declarator/function_declarator wrappers
    rather than exposed as a clean named field."""
    if node is None:
        return None
    for n in iter_nodes(node):
        if n.type in identifier_types:
            return node_text(n, source_bytes)
    return None
