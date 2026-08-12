"""
feature_extraction.py
----------------------
Builds node-level and graph-level features for CodeGraph AI's GraphSAGE
pipeline, given the node/edge JSON schema you're exporting:

    nodes: [{id, type, label, filepath, language, analysed, repo_url}, ...]
    edges: [{id, source, target, type}, ...]

Feature groups produced per node:
  1. Structural / centrality  (degree, pagerank, betweenness, closeness, clustering)
  2. Import / call counts     (per EDGE_TYPES, both incoming and outgoing)
  3. Folder / path features   (depth, parent folder hash, is_root, extension)
  4. Categorical metadata     (node type one-hot, language one-hot, analysed flag)
  5. Lightweight text embedding (hashing-trick over label + filepath tokens)

IMPORTANT: all categorical vocabularies (NODE_TYPES, LANGUAGES, EDGE_TYPES)
are FIXED below rather than inferred from the data. This is deliberate —
inferring them per-repo is what caused your earlier dynamic feat_dim
instability across training runs. Add new categories to these lists as you
encounter them; anything unseen falls into an "OTHER" bucket instead of
silently changing the feature width.
"""

import json
import hashlib
import numpy as np
import networkx as nx
from pathlib import PurePosixPath

# ---------------------------------------------------------------------------
# Fixed vocabularies — extend these deliberately, never infer at runtime
# ---------------------------------------------------------------------------
NODE_TYPES = ["File", "Class", "Function", "Variable", "Module", "OTHER"]
LANGUAGES = [
    "Python", "Shell", "JavaScript", "TypeScript", "C", "C++",
    "Java", "Go", "Rust", "R", "HTML", "CSS", "JSON", "OTHER",
]
EDGE_TYPES = ["DEFINES", "IMPORTS", "CALLS", "INHERITS", "USES", "OTHER"]

EMBED_DIM = 32  # fixed-width hashing-trick text embedding

STRUCTURAL_FEATURE_NAMES = [
    "in_degree", "out_degree", "total_degree",
    "pagerank", "betweenness", "closeness", "clustering",
]


def _bucket(value, vocab):
    return value if value in vocab else "OTHER"


def _one_hot(value, vocab):
    vec = np.zeros(len(vocab), dtype=np.float32)
    vec[vocab.index(_bucket(value, vocab))] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Loading + graph construction
# ---------------------------------------------------------------------------
def load_graph_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["nodes"], data["edges"]


def build_networkx_graph(nodes, edges):
    """MultiDiGraph keeps duplicate edge types (e.g. two CALLS) distinct."""
    G = nx.MultiDiGraph()
    for n in nodes:
        G.add_node(n["id"], **n)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if src not in G or tgt not in G:
            # guards against the "silent data loss" issue — log instead of dropping silently
            print(f"[feature_extraction] WARNING: edge {e.get('id')} references missing node "
                  f"(src_in_graph={src in G}, tgt_in_graph={tgt in G}) — skipping")
            continue
        G.add_edge(src, tgt, key=e.get("id"), type=e.get("type"))
    return G


# ---------------------------------------------------------------------------
# 1. Structural / centrality features
# ---------------------------------------------------------------------------
def compute_centrality_features(G):
    simple_G = nx.DiGraph()
    simple_G.add_nodes_from(G.nodes())
    simple_G.add_edges_from(G.edges())

    in_deg = dict(simple_G.in_degree())
    out_deg = dict(simple_G.out_degree())

    try:
        pagerank = nx.pagerank(simple_G, alpha=0.85, max_iter=200)
    except Exception:
        pagerank = {n: 0.0 for n in simple_G.nodes()}

    n_nodes = simple_G.number_of_nodes()
    if n_nodes > 5000:
        # betweenness is O(V*E); approximate on large graphs (e.g. TensorFlow-scale, 18K+ nodes)
        k = min(500, n_nodes)
        betweenness = nx.betweenness_centrality(simple_G, k=k, normalized=True, seed=42)
    else:
        try:
            betweenness = nx.betweenness_centrality(simple_G, normalized=True)
        except Exception:
            betweenness = {n: 0.0 for n in simple_G.nodes()}

    try:
        closeness = nx.closeness_centrality(simple_G)
    except Exception:
        closeness = {n: 0.0 for n in simple_G.nodes()}

    undirected = simple_G.to_undirected()
    try:
        clustering = nx.clustering(undirected)
    except Exception:
        clustering = {n: 0.0 for n in simple_G.nodes()}

    feats = {}
    for n in G.nodes():
        feats[n] = {
            "in_degree": float(in_deg.get(n, 0)),
            "out_degree": float(out_deg.get(n, 0)),
            "total_degree": float(in_deg.get(n, 0) + out_deg.get(n, 0)),
            "pagerank": float(pagerank.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "closeness": float(closeness.get(n, 0.0)),
            "clustering": float(clustering.get(n, 0.0)),
        }
    return feats


# ---------------------------------------------------------------------------
# 2. Import / call counts, split by edge type and direction
# ---------------------------------------------------------------------------
def compute_edge_type_counts(G):
    """For each node: count of outgoing and incoming edges per EDGE_TYPES bucket."""
    counts = {n: {f"out_{t}": 0.0 for t in EDGE_TYPES} for n in G.nodes()}
    for n in G.nodes():
        counts[n].update({f"in_{t}": 0.0 for t in EDGE_TYPES})

    for u, v, data in G.edges(data=True):
        etype = _bucket(data.get("type"), EDGE_TYPES)
        counts[u][f"out_{etype}"] += 1.0
        counts[v][f"in_{etype}"] += 1.0

    return counts


EDGE_COUNT_FEATURE_NAMES = (
    [f"out_{t}" for t in EDGE_TYPES] + [f"in_{t}" for t in EDGE_TYPES]
)


# ---------------------------------------------------------------------------
# 3. Folder / path features
# ---------------------------------------------------------------------------
def compute_path_features(node_attrs):
    filepath = node_attrs.get("filepath") or ""
    p = PurePosixPath(filepath)
    parts = p.parts
    depth = len(parts) - 1 if parts else 0
    is_root = 1.0 if depth <= 0 else 0.0
    extension = p.suffix.lstrip(".").lower()
    parent = str(p.parent) if depth > 0 else ""

    # stable hashed bucket for parent folder identity (fixed width, no dynamic vocab)
    parent_hash = int(hashlib.md5(parent.encode("utf-8")).hexdigest(), 16) % 997 / 997.0
    ext_hash = int(hashlib.md5(extension.encode("utf-8")).hexdigest(), 16) % 101 / 101.0

    return {
        "path_depth": float(depth),
        "is_root": is_root,
        "parent_folder_hash": parent_hash,
        "extension_hash": ext_hash,
    }


PATH_FEATURE_NAMES = ["path_depth", "is_root", "parent_folder_hash", "extension_hash"]


# ---------------------------------------------------------------------------
# 4. Categorical metadata (fixed-vocab one-hots)
# ---------------------------------------------------------------------------
def compute_categorical_features(node_attrs):
    type_oh = _one_hot(node_attrs.get("type"), NODE_TYPES)
    lang_oh = _one_hot(node_attrs.get("language"), LANGUAGES)
    analysed = np.array([1.0 if node_attrs.get("analysed") else 0.0], dtype=np.float32)
    return np.concatenate([type_oh, lang_oh, analysed])


CATEGORICAL_FEATURE_NAMES = (
    [f"type_{t}" for t in NODE_TYPES] + [f"lang_{l}" for l in LANGUAGES] + ["analysed"]
)


# ---------------------------------------------------------------------------
# 5. Lightweight text embedding (hashing trick — no external model download
#    needed; swap in sentence-transformers later if you want richer semantics)
# ---------------------------------------------------------------------------
def _hash_embed(text, dim=EMBED_DIM):
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    tokens = [t for t in text.replace("/", " ").replace("_", " ").replace(".", " ").split() if t]
    for tok in tokens:
        h = int(hashlib.md5(tok.lower().encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def compute_text_embedding(node_attrs):
    text = f"{node_attrs.get('label', '')} {node_attrs.get('filepath', '')}"
    return _hash_embed(text)


TEXT_EMBED_FEATURE_NAMES = [f"embed_{i}" for i in range(EMBED_DIM)]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
ALL_FEATURE_NAMES = (
    STRUCTURAL_FEATURE_NAMES
    + EDGE_COUNT_FEATURE_NAMES
    + PATH_FEATURE_NAMES
    + CATEGORICAL_FEATURE_NAMES
    + TEXT_EMBED_FEATURE_NAMES
)


def build_node_features(nodes, edges):
    """
    Returns:
        node_ids: list[str] in fixed order
        feature_matrix: np.ndarray of shape (num_nodes, len(ALL_FEATURE_NAMES))
        feature_names: list[str] matching column order
    """
    G = build_networkx_graph(nodes, edges)
    centrality = compute_centrality_features(G)
    edge_counts = compute_edge_type_counts(G)

    node_ids = list(G.nodes())
    rows = []
    for nid in node_ids:
        attrs = G.nodes[nid]

        struct_vec = np.array(
            [centrality[nid][f] for f in STRUCTURAL_FEATURE_NAMES], dtype=np.float32
        )
        edge_vec = np.array(
            [edge_counts[nid][f] for f in EDGE_COUNT_FEATURE_NAMES], dtype=np.float32
        )
        path_feats = compute_path_features(attrs)
        path_vec = np.array([path_feats[f] for f in PATH_FEATURE_NAMES], dtype=np.float32)
        cat_vec = compute_categorical_features(attrs)
        embed_vec = compute_text_embedding(attrs)

        row = np.concatenate([struct_vec, edge_vec, path_vec, cat_vec, embed_vec])
        rows.append(row)

    feature_matrix = np.stack(rows, axis=0) if rows else np.zeros((0, len(ALL_FEATURE_NAMES)))
    assert feature_matrix.shape[1] == len(ALL_FEATURE_NAMES), (
        f"feat_dim mismatch: got {feature_matrix.shape[1]}, expected {len(ALL_FEATURE_NAMES)}"
    )
    return node_ids, feature_matrix, ALL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Optional: convert straight to a PyTorch Geometric Data object
# ---------------------------------------------------------------------------
def to_pyg_data(nodes, edges, label_map=None):
    """
    label_map: optional dict {node_id: np.ndarray label_vector} for supervised training.
    Requires torch and torch_geometric installed.
    """
    import torch
    from torch_geometric.data import Data

    node_ids, feature_matrix, feature_names = build_node_features(nodes, edges)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    src_list, tgt_list = [], []
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in id_to_idx and t in id_to_idx:
            src_list.append(id_to_idx[s])
            tgt_list.append(id_to_idx[t])

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    x = torch.tensor(feature_matrix, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)
    data.node_ids = node_ids
    data.feature_names = feature_names

    if label_map is not None:
        y_rows = [label_map.get(nid, np.zeros(len(next(iter(label_map.values()))))) for nid in node_ids]
        data.y = torch.tensor(np.stack(y_rows), dtype=torch.float32)

    return data


