"""
edge_label_extraction.py
-------------------------
Derives File -> File "dataflow" edges from the Function/Class-level graph
built by graph.py, and produces train/val/test edge splits + negative
sampling helpers for link prediction, all attached to the same
torch_geometric Data object that Label_extraction.py builds for node
classification -- so the GNN encoder trains jointly on both tasks.

Derivation:
  A directed File_A -> File_B edge exists if some Function/Class defined
  in File_A has a CALLS or FLOWS_TO edge (in the unified graph) to a
  Function/Class defined in File_B, A != B. Multiple underlying
  Function-Function edges between the same file pair collapse into one
  File-File edge; the edge label is MULTI-LABEL over EDGE_TYPE_NAMES,
  since a given file pair can genuinely be both a call relationship and
  a data-flow relationship at once. Edges landing on "External" nodes
  are dropped -- there's no second file to attribute them to.

Adds to the existing `data` object (in place):
    data.file_edge_index        LongTensor [2, E]   node-index pairs (into data.x rows)
    data.file_edge_type         FloatTensor [E, T]   multi-label edge type target
    data.file_edge_train_mask   BoolTensor [E]
    data.file_edge_val_mask     BoolTensor [E]
    data.file_edge_test_mask    BoolTensor [E]

Also returns FIXED (seeded) negative edge pairs for val/test -- existence-
head evaluation needs to be stable across epochs and across runs. Training
negatives are instead resampled fresh every step (see model.py), which is
standard practice for link prediction and gives the existence head more
varied negative examples over the course of training.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch

from graph import file_id

SPLIT_SEED = 42  # same seed Label_extraction.py's stratified_split uses

# Fixed vocabulary, mirrors graph.py's structural edge types that can
# meaningfully carry a cross-file relationship. DEFINES/ASSIGNS never
# cross files by construction (a function is always assigned-to within
# its own file), and INHERITS is a structural relationship rather than
# a dataflow one -- both are deliberately excluded here.
EDGE_TYPE_NAMES = ["CALLS", "FLOWS_TO"]

# How many negative (non-edge) file pairs to sample per positive edge,
# for the existence head. 1:1 is a common default for link prediction;
# raise this if the existence head's precision looks inflated by an
# over-easy negative distribution once you see real results.
NEG_RATIO = 1


def _file_owner_index(nodes: list[dict]) -> dict[str, str]:
    """Maps every Class/Function node id -> the File node id that
    DEFINES it, by rebuilding graph.py's file_id() from each node's own
    repo_url/source_file attrs (cheaper than walking DEFINES edges, and
    robust even if DEFINES edges were filtered out upstream for some
    other reason)."""
    file_ids = {n["id"] for n in nodes if n.get("type") == "File"}
    owner = {}
    for n in nodes:
        if n.get("type") in ("Function", "Class"):
            repo = n.get("repo_url")
            sf = n.get("source_file")
            if repo is None or sf is None:
                continue
            fid = file_id(repo, sf)
            if fid in file_ids:
                owner[n["id"]] = fid
    return owner


def derive_file_dataflow_edges(nodes: list[dict], edges: list[dict]) -> dict:
    """Projects Function/Class-level CALLS and FLOWS_TO edges up to the
    File level. Returns {(file_u, file_v): {"CALLS": count, "FLOWS_TO": count}}.
    Self-loops (file_u == file_v) are dropped -- that's intra-file, not
    "between files". Edges into External nodes are dropped -- there's no
    owning file to attribute them to.
    """
    owner = _file_owner_index(nodes)
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for e in edges:
        etype = e.get("type")
        if etype not in EDGE_TYPE_NAMES:
            continue
        src, tgt = e.get("source"), e.get("target")
        file_u = owner.get(src)
        file_v = owner.get(tgt)
        if file_u is None or file_v is None:
            continue  # target is External, or source isn't a Function/Class
        if file_u == file_v:
            continue  # intra-file, not "between files"
        agg[(file_u, file_v)][etype] += 1

    return {k: dict(v) for k, v in agg.items()}


def _stratified_edge_split(type_sets: list[frozenset],
                            train_frac=0.7, val_frac=0.15, seed=SPLIT_SEED):
    """Same stratified-split shape as Label_extraction.stratified_split,
    but stratifying on each edge's *set* of types (e.g. {"CALLS"} vs
    {"CALLS","FLOWS_TO"}) instead of a single argmax label, since edges
    here are multi-label rather than single-label."""
    rng = random.Random(seed)
    buckets: dict[frozenset, list[int]] = defaultdict(list)
    for row, types in enumerate(type_sets):
        buckets[types].append(row)

    train_idx, val_idx, test_idx = [], [], []
    for _, rows in buckets.items():
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

    return train_idx, val_idx, test_idx


def _mask_from_indices(num_edges: int, indices: list[int]):
    mask = torch.zeros(num_edges, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def sample_negative_edges(num_nodes: int, existing_pairs: set[tuple[int, int]],
                           num_samples: int, seed: int | None = None) -> torch.Tensor:
    """Samples `num_samples` (u, v) index pairs, u != v, that are NOT in
    `existing_pairs` (the *directed* set of real edges -- positives from
    ALL splits, not just train, so a negative can never secretly be a
    held-out positive). Returns LongTensor [2, num_samples]. Call with
    seed=None for fresh negatives every training step; call with a fixed
    seed once for val/test so evaluation is stable across epochs/runs.
    """
    rng = random.Random(seed)
    u_list, v_list = [], []
    tries = 0
    max_tries = max(num_samples * 20, 100)  # generous; graphs are sparse so this is cheap
    while len(u_list) < num_samples and tries < max_tries:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        tries += 1
        if u == v or (u, v) in existing_pairs:
            continue
        u_list.append(u)
        v_list.append(v)
    if len(u_list) < num_samples:
        print(f"[edge_label_extraction] WARNING: could only sample "
              f"{len(u_list)}/{num_samples} negative edges (graph may be "
              f"too dense or too small relative to what was requested) -- "
              f"continuing with what we have.")
    return torch.tensor([u_list, v_list], dtype=torch.long)


def build_edge_labels(data, nodes: list[dict], edges: list[dict]):
    """Attaches file_edge_index / file_edge_type / file_edge_{train,val,test}_mask
    to `data` in place (the same Data object Label_extraction.build_labeled_data
    returned), so the multi-task model trains on one shared object.

    Requires data.node_ids to already be set (Label_extraction.py sets this).

    Returns (data, neg_val_edge_index, neg_test_edge_index) -- fixed
    negative pairs for existence-head evaluation on val/test.
    """
    id_to_idx = {nid: i for i, nid in enumerate(data.node_ids)}
    file_edges = derive_file_dataflow_edges(nodes, edges)

    pair_keys, type_dicts = [], []
    for (fu, fv), type_counts in file_edges.items():
        if fu not in id_to_idx or fv not in id_to_idx:
            continue  # shouldn't happen (file ids came from the same node set), but stay safe
        pair_keys.append((fu, fv))
        type_dicts.append(type_counts)

    num_nodes = data.x.size(0)

    if not pair_keys:
        print("[edge_label_extraction] no file-to-file dataflow edges found "
              "-- edge task will be a no-op for this repo (node task is unaffected).")
        data.file_edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.file_edge_type = torch.zeros((0, len(EDGE_TYPE_NAMES)), dtype=torch.float32)
        data.file_edge_train_mask = torch.zeros(0, dtype=torch.bool)
        data.file_edge_val_mask = torch.zeros(0, dtype=torch.bool)
        data.file_edge_test_mask = torch.zeros(0, dtype=torch.bool)
        empty = torch.zeros((2, 0), dtype=torch.long)
        return data, empty, empty

    type_sets = [frozenset(td.keys()) for td in type_dicts]
    train_idx, val_idx, test_idx = _stratified_edge_split(type_sets)

    num_edges = len(pair_keys)
    print(f"[edge_label_extraction] stratified split over {num_edges} derived "
          f"file-to-file dataflow edges: train={len(train_idx)} "
          f"val={len(val_idx)} test={len(test_idx)}")

    src_idx = [id_to_idx[fu] for fu, fv in pair_keys]
    tgt_idx = [id_to_idx[fv] for fu, fv in pair_keys]
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    y_type = np.zeros((num_edges, len(EDGE_TYPE_NAMES)), dtype=np.float32)
    for row, td in enumerate(type_dicts):
        for t in td:
            col = EDGE_TYPE_NAMES.index(t)
            y_type[row, col] = 1.0
    edge_type = torch.tensor(y_type, dtype=torch.float32)

    data.file_edge_index = edge_index
    data.file_edge_type = edge_type
    data.file_edge_train_mask = _mask_from_indices(num_edges, train_idx)
    data.file_edge_val_mask = _mask_from_indices(num_edges, val_idx)
    data.file_edge_test_mask = _mask_from_indices(num_edges, test_idx)

    # Fixed negatives for val/test existence evaluation. Positives from
    # ALL splits (not just val/test) are excluded so a "negative" can
    # never secretly be a held-out train edge.
    all_positive_pairs = {(u, v) for u, v in zip(src_idx, tgt_idx)}
    val_count = int(data.file_edge_val_mask.sum().item())
    test_count = int(data.file_edge_test_mask.sum().item())
    neg_val = sample_negative_edges(num_nodes, all_positive_pairs, val_count * NEG_RATIO, seed=SPLIT_SEED + 1)
    neg_test = sample_negative_edges(num_nodes, all_positive_pairs, test_count * NEG_RATIO, seed=SPLIT_SEED + 2)

    return data, neg_val, neg_test