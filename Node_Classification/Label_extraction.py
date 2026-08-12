"""
Label_extraction.py
--------------------
Turns a predictions file (the output of your classifier — filepath,
repo_url, predicted_label, confidence, is_confident) into the labelled
data that feature_extraction.py / model.py expect:

    label_map     : {node_id: {predicted_label: 1.0}}   (hard, single-label)
    label_meta    : {"label_names": [...known..., "unknown"],
                      "label_to_idx": {name: idx}}
    data.pt       : PyG Data with x, edge_index, y_soft,
                     train_mask / val_mask / test_mask

Design choices:
  - Only `predicted_label` is used as the training target -- `confidence`
    and `all_labels` are ignored entirely (per your call: you only need
    the single predicted label, not a soft/multi-label distribution).
    A confident record becomes a hard one-hot: {predicted_label: 1.0}.
  - Records with is_confident == False are NOT used as ground truth --
    they get an empty label (treated as "unknown") and are excluded
    from train/val/test masks.
  - Node ids are built with graph.py's `file_id(repo_url, filepath)`
    so they line up with the "File" nodes produced by build_unified_graph.
  - The graph/feature side (x) can come from EITHER:
      (a) an already-built unified graph JSON: {"nodes": [...], "edges": [...]}
      (b) the raw multi-key analysis output (file_index, class_diagram,
          function_call_flow, variables_per_function, data_flow, ...) --
          in that case this script runs graph.py's build_unified_graph()
          on it first, so feature_extraction.build_node_features() always
          gets the nodes/edges shape it expects.

To run: just hit "Run" in your editor (no flags needed). It auto-finds a
predictions*.json and a graph-source json (matching *graph*.json or
output*.json) in the current folder. If you keep files elsewhere, or have
more than one match, edit the two paths in the CONFIG block below instead
of passing CLI args.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

from graph import file_id, build_unified_graph, graph_to_json
from feature_extraction import build_node_features

SPLIT_SEED = 42  # matches the seed model.py assumes was used upstream
UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# CONFIG — leave as None to auto-detect the files in the current folder.
# Set either one explicitly if you keep files elsewhere or have more than
# one match (auto-detect will stop and tell you which files it found).
# ---------------------------------------------------------------------------
PREDICTIONS_PATH = None   # e.g. "predictions_2b51f54434bc4e0ab257d13ec99704d1.json"
GRAPH_PATH = None         # e.g. "unified_graph.json"
OUT_DATA_PATH = "data.pt"
OUT_META_PATH = "label_meta.json"
OUT_LABEL_MAP_PATH = "label_map.json"


def _auto_find(patterns, kind: str, explicit: str | None) -> str:
    """Resolves a file path: uses `explicit` if given, otherwise globs
    each pattern in `patterns` (a str or list of str) in the current
    directory and merges the results. Fails loudly (not silently
    guessing) if zero or multiple matches are found, since guessing
    wrong here would silently join the wrong graph/predictions pair."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"[label_extraction] {kind} path '{explicit}' does not exist.")
        return explicit

    if isinstance(patterns, str):
        patterns = [patterns]
    matches = sorted({m for p in patterns for m in glob.glob(p)})

    if len(matches) == 1:
        print(f"[label_extraction] auto-detected {kind}: {matches[0]}")
        return matches[0]
    if len(matches) == 0:
        sys.exit(
            f"[label_extraction] couldn't find a {kind} file matching "
            f"{patterns} in {os.getcwd()}. Set {kind.upper()}_PATH at the "
            f"top of label_extraction.py to the exact file."
        )
    sys.exit(
        f"[label_extraction] found {len(matches)} possible {kind} files: "
        f"{matches}. Set {kind.upper()}_PATH at the top of label_extraction.py "
        f"to the one you want."
    )


# ---------------------------------------------------------------------------
# 0. graph source (x side) -- accepts either a raw analysis-output json
#    (file_index / class_diagram / function_call_flow / ...) or an
#    already-built unified graph json (nodes/edges).
# ---------------------------------------------------------------------------
def load_graph_source(path: str) -> tuple[list, list]:
    with open(path, "r") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "nodes" in raw and "edges" in raw:
        print(f"[label_extraction] {path} is an already-built unified graph "
              f"({len(raw['nodes'])} nodes, {len(raw['edges'])} edges)")
        return raw["nodes"], raw["edges"]

    if isinstance(raw, dict) and "file_index" in raw:
        print(f"[label_extraction] {path} is raw analysis output -- "
              f"building the unified graph with graph.py first")
        G = build_unified_graph(raw, include_data_flow=True, include_variables=False)
        gj = graph_to_json(G)
        print(f"[label_extraction] built graph: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")
        return gj["nodes"], gj["edges"]

    sys.exit(
        f"[label_extraction] '{path}' doesn't look like either a unified "
        f"graph (nodes/edges) or raw analysis output (file_index/...). "
        f"Check GRAPH_PATH."
    )


# ---------------------------------------------------------------------------
# 1. predictions.json -> label_map
# ---------------------------------------------------------------------------
def load_predictions(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def discover_label_vocab(predictions: list[dict]) -> list[str]:
    """Collects every predicted_label seen into a sorted, deterministic
    vocabulary. `all_labels`/`confidence` are intentionally ignored --
    only the single predicted_label is used as a training target."""
    labels = {rec["predicted_label"] for rec in predictions if rec.get("predicted_label")}
    return sorted(labels)


def build_label_map(predictions: list[dict]) -> tuple[dict, list[str]]:
    """Returns (label_map, known_label_names).

    label_map: {node_id: {predicted_label: 1.0}} -- a hard one-hot on
    the single predicted_label for confident records, or {} (treated as
    unknown) for everything else. `confidence` and `all_labels` are not
    used as targets.
    """
    known_label_names = discover_label_vocab(predictions)

    label_map: dict[str, dict[str, float]] = {}
    dropped_unconfident = 0
    seen_ids = set()

    for rec in predictions:
        nid = file_id(rec["repo_url"], rec["filepath"])

        if nid in seen_ids:
            # Same file appearing twice in the predictions file (e.g. a
            # re-run) -- keep the first occurrence rather than silently
            # letting a later duplicate overwrite it.
            continue
        seen_ids.add(nid)

        if not rec.get("is_confident", False) or not rec.get("predicted_label"):
            dropped_unconfident += 1
            label_map[nid] = {}  # unknown -- no training target
            continue

        label_map[nid] = {rec["predicted_label"]: 1.0}

    print(f"[label_extraction] {len(predictions)} predictions -> "
          f"{len(label_map)} unique file nodes "
          f"({dropped_unconfident} marked unknown / low-confidence)")
    return label_map, known_label_names


# ---------------------------------------------------------------------------
# 2. label_map -> y_soft matrix aligned with a fixed node_id order
# ---------------------------------------------------------------------------
def build_y_soft(node_ids: list[str], label_map: dict, known_label_names: list[str]) -> np.ndarray:
    """Dense (num_nodes, num_known_classes) soft-target matrix. Nodes
    absent from label_map (non-File nodes, or File nodes the predictions
    file never covered) get an all-zero row, same as an "unknown" node --
    they simply won't be pulled into train/val/test masks either."""
    idx = {name: i for i, name in enumerate(known_label_names)}
    y = np.zeros((len(node_ids), len(known_label_names)), dtype=np.float32)
    for row, nid in enumerate(node_ids):
        scores = label_map.get(nid)
        if not scores:
            continue
        for label, conf in scores.items():
            col = idx.get(label)
            if col is not None:
                y[row, col] = conf
    return y


# ---------------------------------------------------------------------------
# 3. stratified train/val/test split over the *labelled* (known) nodes
# ---------------------------------------------------------------------------
def stratified_split(node_ids: list[str], label_map: dict, known_label_names: list[str],
                      train_frac=0.7, val_frac=0.15, seed=SPLIT_SEED):
    """Splits only nodes that actually have a confident label (non-empty
    entry in label_map); everything else (unmatched / unknown / non-File
    nodes) is excluded from all three masks so it never leaks into
    training or evaluation. Stratifies on each node's argmax label so
    rare classes still get representation in val/test instead of
    landing entirely in train by chance.
    """
    rng = random.Random(seed)
    idx = {name: i for i, name in enumerate(known_label_names)}

    buckets: dict[str, list[int]] = defaultdict(list)
    for row, nid in enumerate(node_ids):
        scores = label_map.get(nid)
        if not scores:
            continue
        top_label = max(scores.items(), key=lambda kv: kv[1])[0]
        if top_label not in idx:
            continue
        buckets[top_label].append(row)

    train_idx, val_idx, test_idx = [], [], []
    for label, rows in buckets.items():
        rows = rows[:]
        rng.shuffle(rows)
        n = len(rows)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else n
        n_val = max(1, int(round(n * val_frac))) if n - n_train >= 2 else max(0, n - n_train)
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train_idx.extend(rows[:n_train])
        val_idx.extend(rows[n_train:n_train + n_val])
        test_idx.extend(rows[n_train + n_val:])

    total_labelled = sum(len(r) for r in buckets.values())
    print(f"[label_extraction] stratified split over {total_labelled} confidently-labelled "
          f"file nodes across {len(buckets)} classes: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    return train_idx, val_idx, test_idx


def _mask_from_indices(num_nodes: int, indices: list[int]):
    import torch
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


# ---------------------------------------------------------------------------
# 4. end-to-end: graph.json + predictions.json -> data.pt + label_meta.json
# ---------------------------------------------------------------------------
def build_labeled_data(graph_json_path: str, predictions_path: str,
                        out_data_path: str = "data.pt",
                        out_meta_path: str = "label_meta.json",
                        out_label_map_path: str | None = "label_map.json"):
    import torch
    from torch_geometric.data import Data

    nodes, edges = load_graph_source(graph_json_path)
    predictions = load_predictions(predictions_path)

    label_map, known_label_names = build_label_map(predictions)
    label_names = known_label_names + [UNKNOWN]
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    if out_label_map_path:
        with open(out_label_map_path, "w") as f:
            json.dump(label_map, f, indent=2)

    node_ids, feature_matrix, feature_names = build_node_features(nodes, edges)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    src_list, tgt_list = [], []
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in id_to_idx and t in id_to_idx:
            src_list.append(id_to_idx[s])
            tgt_list.append(id_to_idx[t])

    y_soft = build_y_soft(node_ids, label_map, known_label_names)
    train_idx, val_idx, test_idx = stratified_split(node_ids, label_map, known_label_names)

    data = Data(
        x=torch.tensor(feature_matrix, dtype=torch.float32),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )
    data.y_soft = torch.tensor(y_soft, dtype=torch.float32)
    data.train_mask = _mask_from_indices(len(node_ids), train_idx)
    data.val_mask = _mask_from_indices(len(node_ids), val_idx)
    data.test_mask = _mask_from_indices(len(node_ids), test_idx)
    data.node_ids = node_ids
    data.feature_names = feature_names

    torch.save(data, out_data_path)
    with open(out_meta_path, "w") as f:
        json.dump({"label_names": label_names, "label_to_idx": label_to_idx}, f, indent=2)

    print(f"[label_extraction] wrote {out_data_path} "
          f"(x={tuple(data.x.shape)}, y_soft={tuple(data.y_soft.shape)}) "
          f"and {out_meta_path} (label_names={label_names})")

    return data, {"label_names": label_names, "label_to_idx": label_to_idx}


def build_graph_data(graph_json_path: str, out_data_path: str | None = None):
    """Edge-only counterpart to build_labeled_data(): builds a PyG Data
    object with just x / edge_index / node_ids / feature_names from a
    graph JSON, with NO predictions.json and no y_soft / train/val/test
    node masks. Use this when you only care about the File->File
    dataflow edge task (existence + type, see edge_label_extraction.py)
    and don't need node-level architecture-role labels at all.

    build_edge_labels() only needs data.node_ids and data.x to be set --
    both are populated here, so its output can be attached directly.
    """
    import torch
    from torch_geometric.data import Data

    nodes, edges = load_graph_source(graph_json_path)
    node_ids, feature_matrix, feature_names = build_node_features(nodes, edges)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    src_list, tgt_list = [], []
    for e in edges:
        s, t = e.get("source"), e.get("target")
        if s in id_to_idx and t in id_to_idx:
            src_list.append(id_to_idx[s])
            tgt_list.append(id_to_idx[t])

    data = Data(
        x=torch.tensor(feature_matrix, dtype=torch.float32),
        edge_index=torch.tensor([src_list, tgt_list], dtype=torch.long),
    )
    data.node_ids = node_ids
    data.feature_names = feature_names

    print(f"[label_extraction] built edge-only data (x={tuple(data.x.shape)}, "
          f"no labels, no node masks)")

    if out_data_path:
        torch.save(data, out_data_path)
        print(f"[label_extraction] wrote {out_data_path}")

    return data


def main():
    predictions_path = _auto_find("predictions*.json", "predictions", PREDICTIONS_PATH)

    graph_patterns = ["*graph*.json", "output*.json"]
    if GRAPH_PATH is None:
        # don't let the predictions file itself accidentally match a
        # graph pattern (e.g. if someone names it "output_predictions.json")
        graph_matches = sorted({
            m for p in graph_patterns for m in glob.glob(p) if m != predictions_path
        })
        if len(graph_matches) == 1:
            graph_path = graph_matches[0]
            print(f"[label_extraction] auto-detected graph: {graph_path}")
        elif len(graph_matches) == 0:
            sys.exit(
                f"[label_extraction] couldn't find a graph-source file matching "
                f"{graph_patterns} in {os.getcwd()}. Set GRAPH_PATH at the top "
                f"of label_extraction.py to the exact file."
            )
        else:
            sys.exit(
                f"[label_extraction] found {len(graph_matches)} possible graph "
                f"files: {graph_matches}. Set GRAPH_PATH at the top of "
                f"label_extraction.py to the one you want."
            )
    else:
        graph_path = _auto_find(graph_patterns, "graph", GRAPH_PATH)

    build_labeled_data(
        graph_json_path=graph_path,
        predictions_path=predictions_path,
        out_data_path=OUT_DATA_PATH,
        out_meta_path=OUT_META_PATH,
        out_label_map_path=OUT_LABEL_MAP_PATH,
    )