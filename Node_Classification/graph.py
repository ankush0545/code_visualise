"""
graph_builder.py — Phase 3: unified code graph.

Merges file_index, class_diagram, function_call_flow, variables_per_function,
and data_flow (all keyed by repo_url + source_file) into a single typed
NetworkX MultiDiGraph.

Node types : File, Class, Function, Variable, External
Edge types : DEFINES     (File -> Class | Function)
             HAS_METHOD   (Class -> Function)
             INHERITS     (Class -> Class | External)
             CALLS        (Function -> Function | External)
             ASSIGNS      (Function -> Variable)
             FLOWS_TO     (Variable -> Function | External)   [data_flow]

Usage:
    python graph_builder.py merged_input.json
    (reads merged_input.json, then overwrites it in place with the unified graph)
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

# --------------------------------------------------------------------------
# ID scheme — this is the join key that makes "merging" possible.
# Every node gets a deterministic id built from (repo_url, source_file, name).
# --------------------------------------------------------------------------

def file_id(repo_url: str, filepath: str) -> str:
    return f"file::{repo_url}::{filepath}"


def class_id(repo_url: str, source_file: str, class_name: str) -> str:
    return f"class::{repo_url}::{source_file}::{class_name}"


def func_id(repo_url: str, source_file: str, func_name: str) -> str:
    return f"func::{repo_url}::{source_file}::{func_name}"


def var_id(repo_url: str, source_file: str, func_name: str, var_name: str) -> str:
    # variables are scoped to the function they live in, not global
    return f"var::{repo_url}::{source_file}::{func_name}::{var_name}"


def external_id(name: str) -> str:
    # unresolved symbol (stdlib call, third-party import, dynamic dispatch, etc.)
    return f"external::{name}"


_CALL_STRIP_RE = re.compile(r"\(.*\)$")  # strip trailing "(...)" if present


def normalize_call_name(raw: str) -> str:
    """'self.headers.get' -> 'get'; 'int' -> 'int'; strips call parens."""
    raw = _CALL_STRIP_RE.sub("", raw).strip()
    return raw.split(".")[-1] if raw else raw


def normalize_method_sig(sig: str) -> str:
    """'_read_body()' -> '_read_body'"""
    return sig.split("(")[0].strip()


# --------------------------------------------------------------------------
# Index: fast lookups needed to resolve string references into node ids
# --------------------------------------------------------------------------

@dataclass
class RepoIndex:
    # (repo_url, source_file, func_name) -> node id   [exact match]
    func_by_file: dict = field(default_factory=dict)
    # repo_url -> {func_name -> [node_ids]}            [same-repo, any file]
    func_by_repo: dict = field(default_factory=dict)
    # repo_url -> {class_name -> node_id}
    class_by_repo: dict = field(default_factory=dict)


def build_index(data: dict) -> RepoIndex:
    idx = RepoIndex()
    for entry in data.get("function_call_flow", []):
        repo, sf, fn = entry["repo_url"], entry["source_file"], entry["function"]
        nid = func_id(repo, sf, fn)
        idx.func_by_file[(repo, sf, fn)] = nid
        idx.func_by_repo.setdefault(repo, {}).setdefault(fn, []).append(nid)
    for entry in data.get("class_diagram", []):
        repo, sf, cn = entry["repo_url"], entry["source_file"], entry["class"]
        idx.class_by_repo.setdefault(repo, {})[cn] = class_id(repo, sf, cn)
    return idx


def resolve_call(repo: str, source_file: str, raw_call: str, idx: RepoIndex) -> tuple[str, bool]:
    """Returns (node_id, is_external)."""
    name = normalize_call_name(raw_call)
    if not name:
        return external_id(raw_call or "?"), True
    # 1. same-file exact match (cheapest, most reliable)
    same_file = (repo, source_file, name)
    if same_file in idx.func_by_file:
        return idx.func_by_file[same_file], False
    # 2. same-repo, any file (ambiguous if multiple -> take first, flag not needed for MVP)
    candidates = idx.func_by_repo.get(repo, {}).get(name)
    if candidates:
        return candidates[0], False
    # 3. give up -> external node (stdlib / third-party / dynamic)
    return external_id(name), True


def resolve_class(repo: str, base_name: str, idx: RepoIndex) -> tuple[str, bool]:
    base_name = base_name.split(".")[-1].strip()
    nid = idx.class_by_repo.get(repo, {}).get(base_name)
    if nid:
        return nid, False
    return external_id(base_name), True


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_unified_graph(
    data: dict,
    include_data_flow: bool = True,
    include_variables: bool = False,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    idx = build_index(data)

    # ---- Pass 1: nodes -----------------------------------------------
    for entry in data.get("file_index", []):
        repo, fp = entry["repo_url"], entry["filepath"]
        G.add_node(
            file_id(repo, fp),
            type="File",
            name=entry["filename"],
            filepath=fp,
            language=entry.get("language"),
            analysed=entry.get("analysed", False),
            repo_url=repo,
        )

    for entry in data.get("class_diagram", []):
        repo, sf, cn = entry["repo_url"], entry["source_file"], entry["class"]
        G.add_node(
            class_id(repo, sf, cn),
            type="Class",
            name=cn,
            source_file=sf,
            language=entry.get("language"),
            repo_url=repo,
            methods=entry.get("methods", []),
            attributes=entry.get("attributes", []),
        )

    func_params = {}
    func_assigns = {}
    for entry in data.get("variables_per_function", []):
        key = (entry["repo_url"], entry["source_file"], entry["function"])
        func_params[key] = entry.get("params", [])
        func_assigns[key] = entry.get("assigns", [])

    for entry in data.get("function_call_flow", []):
        repo, sf, fn = entry["repo_url"], entry["source_file"], entry["function"]
        key = (repo, sf, fn)
        G.add_node(
            func_id(repo, sf, fn),
            type="Function",
            name=fn,
            source_file=sf,
            language=entry.get("language"),
            repo_url=repo,
            params=func_params.get(key, []),
            assigns=func_assigns.get(key, []),
        )

    # ---- Pass 2: structural edges (File -> Class / Function) ---------
    for entry in data.get("class_diagram", []):
        repo, sf, cn = entry["repo_url"], entry["source_file"], entry["class"]
        fid = file_id(repo, sf)
        if fid in G:
            G.add_edge(fid, class_id(repo, sf, cn), type="DEFINES")

    for entry in data.get("function_call_flow", []):
        repo, sf, fn = entry["repo_url"], entry["source_file"], entry["function"]
        fid = file_id(repo, sf)
        if fid in G:
            G.add_edge(fid, func_id(repo, sf, fn), type="DEFINES")

    # ---- Class -> Function (HAS_METHOD), matched by method name -------
    for entry in data.get("class_diagram", []):
        repo, sf, cn = entry["repo_url"], entry["source_file"], entry["class"]
        cid = class_id(repo, sf, cn)
        for m in entry.get("methods", []):
            mname = normalize_method_sig(m)
            key = (repo, sf, mname)
            target = idx.func_by_file.get(key)
            if target:
                G.add_edge(cid, target, type="HAS_METHOD")

    # ---- Class -> Class (INHERITS) -------------------------------------
    for entry in data.get("class_diagram", []):
        repo, sf, cn = entry["repo_url"], entry["source_file"], entry["class"]
        cid = class_id(repo, sf, cn)
        for base in entry.get("bases", []):
            target, is_ext = resolve_class(repo, base, idx)
            if is_ext and target not in G:
                G.add_node(target, type="External", name=base)
            G.add_edge(cid, target, type="INHERITS")

    # ---- Function -> Function (CALLS) ----------------------------------
    for entry in data.get("function_call_flow", []):
        repo, sf, fn = entry["repo_url"], entry["source_file"], entry["function"]
        src = func_id(repo, sf, fn)
        for raw_call in entry.get("calls", []):
            target, is_ext = resolve_call(repo, sf, raw_call, idx)
            if is_ext and target not in G:
                G.add_node(target, type="External", name=normalize_call_name(raw_call))
            G.add_edge(src, target, type="CALLS", raw=raw_call)

    # ---- Function -> Variable -> (target)   [data_flow, optional] -----
    if include_data_flow:
        for entry in data.get("data_flow", []):
            repo, sf, fn = entry["repo_url"], entry["source_file"], entry["function"]
            var = entry["variable"]
            src = func_id(repo, sf, fn)
            if src not in G:
                continue  # orphan record, skip

            if include_variables:
                vid = var_id(repo, sf, fn, var)
                if vid not in G:
                    G.add_node(vid, type="Variable", name=var, source_file=sf, repo_url=repo)
                G.add_edge(src, vid, type="ASSIGNS")
                flow_src = vid
            else:
                flow_src = src  # collapse variable hop, edge straight from function

            passed_to = entry.get("passed_to")
            if passed_to:
                target, is_ext = resolve_call(repo, sf, passed_to, idx)
                if is_ext and target not in G:
                    G.add_node(target, type="External", name=normalize_call_name(passed_to))
                G.add_edge(
                    flow_src, target, type="FLOWS_TO",
                    variable=var, arg_position=entry.get("arg_position"),
                )

    return G


# --------------------------------------------------------------------------
# JSON export — node-link shape, React Flow-friendly (id/source/target keys)
# --------------------------------------------------------------------------

def graph_to_json(G: nx.MultiDiGraph) -> dict:
    """
    Returns {"nodes": [...], "edges": [...]}.
    Node dict: {"id", "type", "label", ...rest of node attrs}
    Edge dict: {"id", "source", "target", "type", ...rest of edge attrs}
    Kept flat (not nested under "data") on purpose — trivial to remap into
    React Flow's {id, type, data} shape in the frontend, but this stays
    usable for non-React-Flow consumers (e.g. loading straight into
    NetworkX again, or a table view) too.
    """
    nodes = []
    for nid, attrs in G.nodes(data=True):
        node = {"id": nid, "type": attrs.get("type", "Unknown"),
                "label": attrs.get("name", nid)}
        node.update({k: v for k, v in attrs.items() if k not in ("type", "name")})
        nodes.append(node)

    edges = []
    for i, (u, v, attrs) in enumerate(G.edges(data=True)):
        edge = {"id": f"e{i}", "source": u, "target": v,
                "type": attrs.get("type", "RELATED_TO")}
        edge.update({k: val for k, val in attrs.items() if k != "type"})
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def write_json(G: nx.MultiDiGraph, path: str):
    with open(path, "w") as f:
        json.dump(graph_to_json(G), f, indent=2)


