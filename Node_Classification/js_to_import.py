"""
Extend your graph_builder.py pipeline to populate `edges` for JS/TS/TSX/JSX
repos, using a regex-based import extractor (no compiled tree-sitter
grammars needed).

Input:  unified_graph.json  -> {"nodes": [...], "edges": []}
        + local checkout of the repo on disk (source_root)
Output: unified_graph.json  -> same shape, but with real edges populated,
        plus a flow_edges.json in the {source, target, flow} shape.
"""

import json
import os
import re
from pathlib import Path

JS_TS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx"}

# Matches:
#   import X from 'path'
#   import {a, b} from "path"
#   import * as X from 'path'
#   import 'path'                (side-effect import)
#   export { a } from 'path'
#   export * from 'path'
#   const x = require('path')
#   import('path')                (dynamic import)
IMPORT_PATTERNS = [
    re.compile(r"""import\s+(?:[\w*{}\s,]+\s+from\s+)?['"](?P<path>[^'"]+)['"]"""),
    re.compile(r"""export\s+(?:[\w*{}\s,]+\s+from\s+)?['"](?P<path>[^'"]+)['"]"""),
    re.compile(r"""require\(\s*['"](?P<path>[^'"]+)['"]\s*\)"""),
    re.compile(r"""import\(\s*['"](?P<path>[^'"]+)['"]\s*\)"""),
]



def extract_js_imports(source_code: str) -> list[str]:
    """Return raw import path strings found in a JS/TS source file."""
    imports = []
    for pattern in IMPORT_PATTERNS:
        imports.extend(m.group("path") for m in pattern.finditer(source_code))
    return imports


def resolve_import_path(importer_filepath: str, import_str: str, all_filepaths: set) -> str | None:
    """
    Resolve an import string to an actual filepath present in the node list.
    Only resolves relative imports (./, ../) — skips bare package imports
    (e.g. 'react', 'lodash') since those aren't files in this repo's graph.
    """
    if not (import_str.startswith(".") or import_str.startswith("/")):
        return None  # external package, not a local file — skip

    importer_dir = os.path.dirname(importer_filepath)
    raw_target = os.path.normpath(os.path.join(importer_dir, import_str))
    raw_target = raw_target.replace(os.sep, "/")

    candidates = [raw_target]
    # try common extensions
    for ext in JS_TS_EXTENSIONS:
        candidates.append(raw_target + ext)
    # try index files if it resolves to a directory
    for ext in JS_TS_EXTENSIONS:
        candidates.append(f"{raw_target}/index{ext}")

    for candidate in candidates:
        if candidate in all_filepaths:
            return candidate
        # also try matching without a leading path fragment mismatch
        if candidate.lstrip("/") in all_filepaths:
            return candidate.lstrip("/")

    return None


def populate_edges(unified_graph_path: str, source_root: str, output_path: str | None = None) -> dict:
    with open(unified_graph_path) as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    filepath_to_id = {n["filepath"]: n["id"] for n in nodes}
    all_filepaths = set(filepath_to_id.keys())

    edges = []
    analysed_count = 0

    for node in nodes:
        filepath = node["filepath"]
        ext = os.path.splitext(filepath)[1]
        if ext not in JS_TS_EXTENSIONS:
            continue

        full_path = os.path.join(source_root, filepath)
        if not os.path.exists(full_path):
            continue

        source_code = Path(full_path).read_text(errors="ignore")
        raw_imports = extract_js_imports(source_code)

        node["analysed"] = True
        analysed_count += 1

        for import_str in raw_imports:
            target_fp = resolve_import_path(filepath, import_str, all_filepaths)
            if not target_fp or target_fp == filepath:
                continue
            edges.append({
                "source": node["id"],
                "target": filepath_to_id[target_fp],
                "type": "IMPORTS",
            })

    # de-duplicate edges
    seen = set()
    unique_edges = []
    for e in edges:
        key = (e["source"], e["target"], e["type"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    graph["edges"] = unique_edges

    out = output_path or unified_graph_path
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Analysed {analysed_count} JS/TS files, found {len(unique_edges)} edges. Wrote {out}")
    return graph


# ---- Convert unified_graph edges into the {source, target, flow} shape ------

ROLE_RULES = [
    (lambda p: "/components/" in p or "/pages/" in p or p.endswith((".jsx", ".tsx")), "FRONTEND"),
    (lambda p: "/hooks/" in p, "FRONTEND"),
    (lambda p: "/api/" in p or "endpoint" in p or "/server/" in p, "API"),
    (lambda p: "/db/" in p or "database" in p or "schema" in p, "DB"),
    (lambda p: True, "OTHER"),
]


def classify_role(filepath: str) -> str:
    fp = filepath.lower()
    for predicate, role in ROLE_RULES:
        if predicate(fp):
            return role
    return "OTHER"


def to_flow_edges(graph: dict) -> list[dict]:
    id_to_node = {n["id"]: n for n in graph["nodes"]}
    flow_edges = []
    for e in graph["edges"]:
        src_node = id_to_node[e["source"]]
        tgt_node = id_to_node[e["target"]]
        src_role = classify_role(src_node["filepath"])
        tgt_role = classify_role(tgt_node["filepath"])
        flow_edges.append({
            "source": src_node["label"],
            "target": tgt_node["label"],
            "flow": f"{src_role}_TO_{tgt_role}",
        })
    return flow_edges


if __name__ == "__main__":
    graph = populate_edges(
        unified_graph_path="unified_graph.json",
        source_root="./repo_cache/ant-design",  # point at your local clone
        output_path="unified_graph_with_edges.json",
    )
    flow_edges = to_flow_edges(graph)
    with open("flow_edges.json", "w") as f:
        json.dump(flow_edges, f, indent=2)
    print(f"Wrote {len(flow_edges)} flow edges to flow_edges.json")

    