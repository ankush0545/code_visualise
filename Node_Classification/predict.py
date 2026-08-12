"""
predict.py
----------
Inference-only script for the edge-only GNN trained by model.py's
modelling_edges() (checkpoint default: finetuned_sage_edges.pt).

Given a repo graph JSON ({"nodes": [...], "edges": [...]}, e.g. anything
under repo_outputs/), this loads the trained encoder + existence head +
type head, rebuilds node features with the SAME pipeline used at training
time (Label_extraction.build_graph_data -> feature_extraction), and scores
candidate File->File pairs for:
  - edge_exists_prob : sigmoid probability a dataflow edge exists
  - predicted_types  : which of EDGE_TYPE_NAMES (CALLS / FLOWS_TO) cross
                        --type-threshold
  - type_probs       : per-type sigmoid probability

No retraining happens here -- this is pure forward-pass inference.

Usage
-----
    # Predict undiscovered File->File relationships in a repo (default mode)
    python predict.py repo_outputs/metaflow.json

    # Sanity-check the model against edges it was trained/evaluated on
    python predict.py repo_outputs/actix-web.json --candidates existing

    # Only keep the 100 most confident new-edge predictions
    python predict.py repo_outputs/actix-web.json --top-k 100

    # Use a GCN checkpoint instead of the default SAGE one
    python predict.py repo_outputs/actix-web.json \
        --model-type gcn --checkpoint finetuned_gcn_edges.pt

    # No path given: scores every *.json file under INPUT_FILEPATH
    python predict.py

    # Prompt for repo graph paths one at a time
    python predict.py --interactive
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import itertools
import json
import random
import torch
from Label_extraction import load_graph_source, build_graph_data
from model import GNNEncoder, EdgeHead, VALID_MODEL_TYPES
from edge_label_extraction import EDGE_TYPE_NAMES, derive_file_dataflow_edges



DEFAULT_CHECKPOINT = "Node_Classification/finetuned_sage_edges.pt"
SEED = 42
INPUT_FILEPATH = "data/repo_outputs"



# Same cache folder main.py writes per-repo PyG data objects to. Reusing
# it (and the same namespaced-by-repo_id filename pattern) means predict.py
# picks up main.py's already-cached data for a repo instead of silently
# overwriting/reading a single shared "data_edges.pt" -- that collision is
# the exact bug main.py's out_data_path namespacing was added to fix, and
# build_graph_data() defaults to that same shared path if you don't pass
# one explicitly.
DATA_CACHE_DIR = "data_cache"




def load_model(checkpoint_path, feat_dim, model_type, hidden_channels,
                out_channels, num_layers, heads):
    """Rebuilds the encoder + two heads with the SAME shapes used in
    model.modelling_edges(), then loads trained weights. Shapes must
    match what the checkpoint was saved with -- if you trained with
    non-default hidden/out channels, pass the same values here."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train a model first "
            f"via main.py (which calls model.modelling_edges()), or point "
            f"--checkpoint at the right file."
        )

    encoder = GNNEncoder(
        in_channels=feat_dim, hidden_channels=hidden_channels,
        out_channels=out_channels, num_layers=num_layers,
        conv_type=model_type, heads=heads,
    )
    exist_head = EdgeHead(out_channels, 1)
    type_head = EdgeHead(out_channels, len(EDGE_TYPE_NAMES))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        encoder.load_state_dict(checkpoint["encoder"])
        exist_head.load_state_dict(checkpoint["exist_head"])
        type_head.load_state_dict(checkpoint["type_head"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' doesn't match the requested "
            f"architecture (model_type={model_type!r}, "
            f"hidden={hidden_channels}, out={out_channels}, "
            f"layers={num_layers}, heads={heads}). Pass the same "
            f"hyperparameters used during training. Original error: {e}"
        ) from e

    encoder.eval()
    exist_head.eval()
    type_head.eval()
    return encoder, exist_head, type_head


def build_candidate_pairs(nodes, edges, node_ids, mode, max_pairs, seed=SEED):
    """Returns a [2, P] LongTensor of node-index pairs (into data.x rows)
    to score, restricted to File nodes (the only node type the edge task
    was trained on).

    mode="existing": only pairs already connected by a derived CALLS/
        FLOWS_TO File->File edge -- useful to check the model against
        ground truth on a repo it has (or resembles) training data for.
    mode="new": every ordered File pair NOT already connected -- actual
        link prediction over undiscovered relationships. Default.
    mode="all": every ordered File pair, known or not.
    """
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    file_ids = [n["id"] for n in nodes if n.get("type") == "File" and n["id"] in id_to_idx]
    file_idx = [id_to_idx[fid] for fid in file_ids]

    existing = derive_file_dataflow_edges(nodes, edges)  # {(fu_id, fv_id): {type: count}}
    existing_pairs_idx = {
        (id_to_idx[fu], id_to_idx[fv])
        for (fu, fv) in existing
        if fu in id_to_idx and fv in id_to_idx
    }

    if mode == "existing":
        pairs = list(existing_pairs_idx)
    else:
        pairs = [
            (u, v) for u, v in itertools.permutations(file_idx, 2)
            if not (mode == "new" and (u, v) in existing_pairs_idx)
        ]

    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(seed)
        print(f"[predict] {len(pairs)} candidate pairs exceeds --max-pairs="
              f"{max_pairs}; randomly sampling down (seed={seed}). "
              f"Pass --max-pairs 0 to disable capping.")
        pairs = rng.sample(pairs, max_pairs)

    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [p[0] for p in pairs]
    tgt = [p[1] for p in pairs]
    return torch.tensor([src, tgt], dtype=torch.long)


@torch.no_grad()
def score_pairs(encoder, exist_head, type_head, data, pair_index, batch_size=4096):
    """Encodes the whole graph once, then scores candidate pairs in
    batches (existence + type heads are cheap MLPs, so batching here is
    just to keep peak memory bounded on very large candidate sets)."""
    z = encoder(data.x, data.edge_index)

    exist_chunks, type_chunks = [], []
    num_pairs = pair_index.size(1)
    for start in range(0, num_pairs, batch_size):
        chunk = pair_index[:, start:start + batch_size]
        exist_logits = exist_head(z, chunk).squeeze(-1)
        type_logits = type_head(z, chunk)
        exist_chunks.append(torch.sigmoid(exist_logits))
        type_chunks.append(torch.sigmoid(type_logits))

    exist_probs = torch.cat(exist_chunks) if exist_chunks else torch.zeros(0)
    type_probs = torch.cat(type_chunks) if type_chunks else torch.zeros((0, len(EDGE_TYPE_NAMES)))
    return exist_probs, type_probs


def score_graph(graph_json_path, encoder, exist_head, type_head, output_path,
                 candidates="new", exist_threshold=0.5, type_threshold=0.5,
                 top_k=None, max_pairs=500_000, data=None):
    """Runs the full predict pipeline for ONE repo graph against an
    already-loaded model, writing results to output_path. This is the
    part that used to be main()'s body -- split out so the expensive
    model load happens exactly once regardless of how many repos get
    scored in a run.

    Every tunable is passed explicitly rather than bundled into an argparse
    Namespace, so this function (and the model it was handed) can be called
    directly -- from a REPL, a loop over repos, or another script -- without
    needing to fake up an `args` object first.

    data: optional pre-built build_graph_data() result. Pass this for the
    very first graph in a run, since main() already had to build it to
    infer feat_dim before load_model() could run; every other call can
    leave this as None and let this function build it.
    """
    nodes, edges = load_graph_source(graph_json_path)
    print(f"[predict] loaded graph: {len(nodes)} nodes, {len(edges)} edges "
          f"({graph_json_path})")


    if data is None:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        # Same feature-building path used at training time (build_graph_data
        # -> feature_extraction.build_node_features), so x lines up with what
        # the checkpoint's encoder expects feature-wise. out_data_path is
        # namespaced by repo_id -- without this, build_graph_data() falls
        # back to a single shared cache file, so predicting on repo B after
        # repo A would silently reuse/overwrite repo A's cached data instead
        # of building repo B's own features (the same collision main.py's
        # training loop hit before it namespaced data_edges_<repo_id>.pt).
        data = build_graph_data(
            graph_json_path=graph_json_path,
            out_data_path=os.path.join(DATA_CACHE_DIR, f"data_edges.pt"),
        )

    pairs_cap = max_pairs if max_pairs and max_pairs > 0 else None
    pair_index = build_candidate_pairs(nodes, edges, data.node_ids, candidates, pairs_cap)
    print(f"[predict] scoring {pair_index.size(1)} candidate File->File "
          f"pair(s) (mode='{candidates}')")

    if pair_index.size(1) == 0:
        print("[predict] no candidate pairs to score -- nothing to write.")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return

    exist_probs, type_probs = score_pairs(encoder, exist_head, type_head, data, pair_index)

    id_to_filepath = {n["id"]: n.get("filepath", n["id"]) for n in nodes}
    node_ids = data.node_ids

    results = []
    for i in range(pair_index.size(1)):
        u_idx, v_idx = pair_index[0, i].item(), pair_index[1, i].item()
        exist_prob = exist_probs[i].item()
        if exist_prob < exist_threshold:
            continue
        u_id, v_id = node_ids[u_idx], node_ids[v_idx]
        t_probs = type_probs[i].tolist()
        predicted_types = [
            EDGE_TYPE_NAMES[j] for j, p in enumerate(t_probs) if p >= type_threshold
        ]
        results.append({
            "source_id": u_id,
            "target_id": v_id,
            "source_filepath": id_to_filepath.get(u_id, u_id),
            "target_filepath": id_to_filepath.get(v_id, v_id),
            "edge_exists_prob": round(exist_prob, 4),
            "predicted_types": predicted_types,
            "type_probs": {name: round(p, 4) for name, p in zip(EDGE_TYPE_NAMES, t_probs)},
        })

    results.sort(key=lambda r: r["edge_exists_prob"], reverse=True)
    if top_k:
        results = results[:top_k]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[predict] wrote {len(results)} predicted edge(s) "
          f"(exist_prob >= {exist_threshold}) to {output_path}\n")




def main():
    parser = argparse.ArgumentParser(
        description="Predict File->File dataflow edges (existence + type) "
                     "with a trained edge-only GNN checkpoint."
    )
    parser.add_argument("graph_json", nargs="*", default=None,
                         help="Path(s) to repo graph JSON(s) to score, e.g. "
                              "repo_outputs/metaflow.json. If omitted and "
                              "--interactive isn't set, every *.json file "
                              f"under '{INPUT_FILEPATH}' is scored.")
    parser.add_argument("--interactive", action="store_true",
                         help="Load the model once, then prompt for repo graph "
                              "paths one at a time (blank line / 'quit' to exit). "
                              "Implied automatically if no graph_json is given.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-type", default="sage", choices=VALID_MODEL_TYPES)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--candidates", choices=["new", "existing", "all"], default="new",
                         help="'new' (default): score undiscovered File pairs only. "
                              "'existing': sanity-check against already-derived edges. "
                              "'all': every ordered File pair, known or not.")
    parser.add_argument("--exist-threshold", type=float, default=0.5,
                         help="Minimum edge_exists_prob to keep in the output.")
    parser.add_argument("--type-threshold", type=float, default=0.5,
                         help="Minimum per-type prob to include in predicted_types.")
    parser.add_argument("--top-k", type=int, default=None,
                         help="Keep only the top-K highest edge_exists_prob predictions.")
    parser.add_argument("--max-pairs", type=int, default=500_000,
                         help="Cap on candidate pairs scored (random-sampled if "
                              "exceeded, for large repos). 0 disables the cap.")
    parser.add_argument("--output", default="predicted_edges.json",
                         help="Output path when scoring a single graph. When "
                              "scoring several (multiple graph_json args, or "
                              "--interactive), each repo instead gets its own "
                              "'<output-stem>_<repo_id>.json' so runs don't "
                              "clobber each other's results.")
    parsed = parser.parse_args()

    # Resolve graph_paths from (in priority order): explicit CLI paths,
    # --interactive with none given yet, or a directory scan fallback so
    # `python predict.py` / master_main.py's no-arg subprocess call still
    # works. Previously this was a for-loop over os.listdir() that
    # OVERWROTE graph_paths every iteration (only the last file survived)
    # and left graph_paths/interactive completely unset -- causing an
    # UnboundLocalError at `len(graph_paths)` below -- whenever the
    # directory was empty or the loop didn't run.
    if parsed.graph_json:
        graph_paths = parsed.graph_json
        interactive = parsed.interactive
    elif parsed.interactive:
        graph_paths = []
        interactive = True
    else:
        if os.path.isdir(INPUT_FILEPATH):
            graph_paths = [
                os.path.join(INPUT_FILEPATH, f)
                for f in os.listdir(INPUT_FILEPATH)
                if f.endswith(".json")
            ]
        else:
            graph_paths = []
        interactive = False

        if not graph_paths:
            print(f"[predict] no graph JSON files found under "
                  f"'{INPUT_FILEPATH}' and none passed on the command line. "
                  f"Pass a path directly, e.g. "
                  f"'python predict.py repo_outputs/metaflow.json', or use "
                  f"--interactive.")
            return

    checkpoint = parsed.checkpoint
    model_type = parsed.model_type
    hidden_channels = parsed.hidden_channels
    out_channels = parsed.out_channels
    num_layers = parsed.num_layers
    heads = parsed.heads
    candidates = parsed.candidates
    exist_threshold = parsed.exist_threshold
    type_threshold = parsed.type_threshold
    top_k = parsed.top_k
    max_pairs = parsed.max_pairs
    output_template = parsed.output

    multi_run = len(graph_paths) > 1 or interactive

    # ---- Load the model ONCE. Everything below reuses these three
    # modules across every graph passed in, and across every interactive
    # iteration -- this is the whole point of this refactor: load_model()/
    # torch.load() used to run again for every repo you wanted predictions
    # on, which dominated wall-clock time when checking several repos back
    # to back.
    #
    # feat_dim comes from build_graph_data on the FIRST graph rather than a
    # hardcoded constant, since it must match whatever feature_extraction.py
    # produced for the checkpoint to load cleanly -- but it's the same
    # feature schema for every repo, so building it once up front and
    # reusing the resulting encoder for later graphs is safe.
    first_graph = graph_paths[0] if graph_paths else None
    if first_graph is None:
        # No positional args: ask for the first path now so we have a
        # graph to infer feat_dim from before the model can be loaded.
        first_graph = input("[predict] path to first repo graph JSON: ").strip()
        if not first_graph:
            print("[predict] no graph provided, exiting.")
            return


    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    first_data = build_graph_data(
        graph_json_path=first_graph,
        out_data_path=os.path.join(DATA_CACHE_DIR, "data_edges.pt"),
    )
    feat_dim = first_data.x.size(1)

    print(f"[predict] loading checkpoint '{checkpoint}' (model_type={model_type})...")
    encoder, exist_head, type_head = load_model(
        checkpoint, feat_dim, model_type, hidden_channels, out_channels, num_layers, heads,
    )
    print("[predict] model loaded -- scoring graphs without reloading.\n")

    os.makedirs("Final_Output", exist_ok=True)

    # Score the first graph using the data we already built, then continue
    # on to the rest of graph_paths (if any) and/or the interactive loop --
    # all against the SAME encoder/exist_head/type_head.
    score_graph(
        first_graph, encoder, exist_head, type_head,
        output_path=f"Final_Output/{os.path.splitext(os.path.basename(first_graph))[0]}_predicted_edges.json",
        candidates=candidates, exist_threshold=exist_threshold,
        type_threshold=type_threshold, top_k=top_k, max_pairs=max_pairs,
        data=first_data,
    )

    for gpath in graph_paths[1:]:
        score_graph(
            gpath, encoder, exist_head, type_head,
            output_path=f"Final_Output/{os.path.splitext(os.path.basename(gpath))[0]}_predicted_edges.json",
            candidates=candidates, exist_threshold=exist_threshold,
            type_threshold=type_threshold, top_k=top_k, max_pairs=max_pairs,
        )

    if interactive:
        print("[predict] interactive mode -- enter another repo graph path, "
              "or blank/'quit' to stop.")
        while True:
            try:
                gpath = input("[predict] repo graph JSON path: ").strip()
            except EOFError:
                break
            if not gpath or gpath.lower() in ("quit", "exit"):
                break
            try:
                score_graph(
                    gpath, encoder, exist_head, type_head,
                    output_path=f"Final_Output/{os.path.splitext(os.path.basename(gpath))[0]}_predicted_edges.json",
                    candidates=candidates, exist_threshold=exist_threshold,
                    type_threshold=type_threshold, top_k=top_k, max_pairs=max_pairs,
                )
            except Exception as e:
                print(f"[predict] error scoring '{gpath}': {e}")


if __name__ == "__main__":
    main()