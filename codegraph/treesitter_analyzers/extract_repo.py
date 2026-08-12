"""
treesitter_analyzers.extract_repo
------------------------------------
Walks a repo checkout and produces the SAME raw-analysis dict shape
your Python/AST pipeline writes into Data/output_<id>.json:

    {
      "file_index":            [...],
      "class_diagram":         [...],
      "function_call_flow":    [...],
      "variables_per_function":[...],
      "data_flow":             [...],
    }

This means it needs ZERO changes to graph.py, Label_extraction.py,
edge_label_extraction.py, or model.py -- it's a drop-in additional
source feeding the same merge_raw_analysis() / build_unified_graph()
pipeline in main.py, for the 6 languages your Python/AST extractor
can't cover (JS, TS, Go, Java, C, C++).

Usage:
    from treesitter_analyzers.extract_repo import extract_repo
    raw = extract_repo("./repo_cache/some-repo", repo_url="https://github.com/...")
    # then merge `raw` into your existing merged dict exactly like any
    # other Data/output_*.json file, and hand it to build_unified_graph()
"""

from __future__ import annotations

import os

from .common import EXTENSION_LANGUAGE_MAP, parse_source
from .class_flow_ts import ClassDiagramAnalyzerTS
from .function_flow_ts import CallFlowAnalyzerTS
from .data_flow_ts import DataFlowAnalyzerTS

# Directories never worth walking into -- keeps large repos (node_modules,
# vendored deps, build output) from silently ballooning file_index and
# skewing the node-role classifier with irrelevant files.
IGNORE_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "target",
    "__pycache__", ".venv", "venv", "third_party", "bazel-bin", "bazel-out",
}


def extract_repo(repo_dir: str, repo_url: str) -> dict:
    file_index = []
    class_diagram = []
    function_call_flow = []
    variables_per_function = []
    data_flow = []

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTENSION_LANGUAGE_MAP:
                continue

            grammar_name, display_lang = EXTENSION_LANGUAGE_MAP[ext]
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_dir).replace(os.sep, "/")

            try:
                with open(full_path, "rb") as f:
                    source = f.read()
            except OSError as e:
                print(f"[extract_repo] WARNING: couldn't read {full_path}: {e}")
                continue

            analysed = False
            try:
                tree = parse_source(source, grammar_name)

                class_analyzer = ClassDiagramAnalyzerTS(grammar_name)
                class_analyzer.analyze(tree, source)

                call_analyzer = CallFlowAnalyzerTS(grammar_name)
                call_analyzer.analyze(tree, source)

                data_analyzer = DataFlowAnalyzerTS(grammar_name)
                data_analyzer.analyze(tree, source)

                analysed = True
            except Exception as e:
                # Parser errors on a single file shouldn't kill the whole
                # repo walk -- log and keep the File node with analysed=False,
                # same graceful-degradation spirit as the rest of the pipeline.
                print(f"[extract_repo] WARNING: failed to analyze {rel_path}: {e}")
                class_analyzer = call_analyzer = data_analyzer = None

            file_index.append({
                "repo_url": repo_url,
                "filepath": rel_path,
                "filename": fname,
                "language": display_lang,
                "analysed": analysed,
            })

            if not analysed:
                continue

            for cls_name, cls_data in class_analyzer.classes.items():
                class_diagram.append({
                    "repo_url": repo_url,
                    "source_file": rel_path,
                    "class": cls_name,
                    "bases": cls_data["bases"],
                    "attributes": cls_data["attributes"],
                    "methods": cls_data["methods"],
                    "language": display_lang,
                })

            calls_by_fn: dict[str, list[str]] = {fn: [] for fn in call_analyzer.function_names}
            for caller, callee in call_analyzer.calls:
                calls_by_fn.setdefault(caller, []).append(callee)
            for fn_name, calls in calls_by_fn.items():
                function_call_flow.append({
                    "repo_url": repo_url,
                    "source_file": rel_path,
                    "function": fn_name,
                    "calls": calls,
                    "language": display_lang,
                })

            for fn_name, entries in data_analyzer.assignments.items():
                params = [e[len("param:"):] for e in entries if e.startswith("param:")]
                assigns = [e for e in entries if not e.startswith("param:")]
                variables_per_function.append({
                    "repo_url": repo_url,
                    "source_file": rel_path,
                    "function": fn_name,
                    "params": params,
                    "assigns": assigns,
                })

            for fn_name, var, passed_to, arg_pos in data_analyzer.data_flows:
                data_flow.append({
                    "repo_url": repo_url,
                    "source_file": rel_path,
                    "function": fn_name,
                    "variable": var,
                    "passed_to": passed_to,
                    "arg_position": arg_pos,
                })

    print(f"[extract_repo] {repo_url}: {len(file_index)} files, "
          f"{len(class_diagram)} classes, {len(function_call_flow)} functions, "
          f"{sum(len(f['calls']) for f in function_call_flow)} calls, "
          f"{len(data_flow)} data-flow edges")

    return {
        "file_index": file_index,
        "class_diagram": class_diagram,
        "function_call_flow": function_call_flow,
        "variables_per_function": variables_per_function,
        "data_flow": data_flow,
    }
