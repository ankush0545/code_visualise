import json
import os
import re
from collections import defaultdict

from graph import build_unified_graph, write_json
from Label_extraction import build_graph_data, load_graph_source
from edge_label_extraction import build_edge_labels
from model import modelling_edges  # edge-only: existence + type heads, no node classifier

# Folder of already-built per-repo graph JSON files, one per repo
# (e.g. actix-web.json, Agent-Reach.json, ...), each shaped
# {"nodes": [...], "edges": [...]}.
UNIFIED_GRAPH_DIR = "codegraph/repo_outputs"

# Where per-repo PyG data objects get cached, namespaced by repo_id so
# repos don't clobber each other's data_edges_<repo_id>.pt file.
DATA_CACHE_DIR = "data_cache"


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def file_extract(folder_path: str):
    """Returns full paths of every .json file directly under folder_path."""
    json_contents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            json_contents.append(os.path.join(folder_path, filename))
    return json_contents


def merge_raw_analysis(file_paths: list[str]) -> dict:
    """For the EARLIER pipeline stage only: merges raw per-repo analyzer
    output (file_index/class_diagram/function_call_flow/... keys) into
    one dict before build_unified_graph(). NOT used in main() below --
    the files in repo_outputs/ are already-built graphs, not raw
    analyzer output, so this step doesn't apply to them. Kept here for
    whenever you need to go from fresh analyzer output to a unified
    graph JSON in the first place."""
    merged: dict[str, list] = defaultdict(list)
    for fp in file_paths:
        data = load_json(fp)
        for key, records in data.items():
            if isinstance(records, list):
                merged[key].extend(records)
    return dict(merged)


def extract_repo_id(filename: str) -> str:
    """Derives a unique id per repo file so checkpoints/outputs/data
    caches get namespaced separately -- reusing the same filenames
    across repos was the checkpoint-collision bug this pipeline hit
    before with per-repo training.

    Files under repo_outputs/ are named "<repo-name>.json" (e.g.
    "actix-web.json", "Agent-Reach.json"), so the id is just the
    filename without the extension. The old "unified_graph_<hex>.json"
    pattern is still matched first in case you point this at that
    naming scheme instead.
    """
    match = re.search(r"unified_graph_([a-f0-9]+)\.json", filename)
    if match:
        return match.group(1)
    # repo_outputs/ naming: "<repo-name>.json" -> "<repo-name>"
    return os.path.splitext(filename)[0]


def main():
    if not os.path.isdir(UNIFIED_GRAPH_DIR):
        raise FileNotFoundError(
            f"[main] '{UNIFIED_GRAPH_DIR}' does not exist (cwd={os.getcwd()}). "
            f"Point UNIFIED_GRAPH_DIR at the folder containing your "
            f"per-repo graph JSON files, or run main.py from the "
            f"right directory."
        )

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    json_files = sorted(f for f in os.listdir(UNIFIED_GRAPH_DIR) if f.endswith(".json"))
    if not json_files:
        raise ValueError(f"[main] no .json files found in '{UNIFIED_GRAPH_DIR}'.")

    all_results = []
    skipped = []

    for filename in json_files:
        file_path = os.path.join(UNIFIED_GRAPH_DIR, filename)
        repo_id = extract_repo_id(filename)

        print(f"\n{'='*70}\nProcessing {filename} (repo_id={repo_id})\n{'='*70}")

        try:
            # Each file here is already a BUILT unified graph
            # ({"nodes": [...], "edges": [...]}), not raw per-repo analysis
            # output -- so it goes straight to load_graph_source(), which
            # auto-detects this shape. Do NOT route it through
            # merge_raw_analysis()/build_unified_graph(): those expect raw
            # file_index/class_diagram/... keys, which an already-built
            # graph doesn't have -- that mismatch is what produced
            # "Nodes: 0 Edges: 0" before.
            nodes, edges = load_graph_source(file_path)
            print(f"[main] {filename}: {len(nodes)} nodes, {len(edges)} edges")

            if len(nodes) == 0:
                print(f"[main] WARNING: {filename} has 0 nodes -- skipping.")
                skipped.append((filename, "0 nodes"))
                continue

            # Edge-only path: no predictions.json / node labels needed at all --
            # we only train on the File->File dataflow edges (existence + type).
            # NOTE: out_data_path is namespaced per repo_id -- previously this
            # was a single hardcoded "data_edges.pt" shared across every repo
            # in the loop, so each iteration silently overwrote the last
            # repo's cached data before modelling_edges() ran on it.
            data = build_graph_data(
                graph_json_path=file_path,
                out_data_path=os.path.join(DATA_CACHE_DIR, f"data_edges_{repo_id}.pt"),
            )

            data, neg_val, neg_test = build_edge_labels(
                data,
                nodes,
                edges,
            )

            if data.file_edge_index.size(1) == 0:
                print(f"[main] WARNING: {filename} has 0 derived File->File "
                      f"dataflow edges -- nothing for modelling_edges() to "
                      f"train on, skipping.")
                skipped.append((filename, "0 dataflow edges"))
                continue

            results = modelling_edges(
                data,
                neg_val,
                neg_test,
                model_type="sage",
            )
            results["repo_id"] = repo_id
            results["source_file"] = filename

            print(results)
            all_results.append(results)

        except Exception as e:
            # Don't let one bad repo file kill the whole batch run --
            # log it, skip it, keep going through the rest of repo_outputs/.
            print(f"[main] ERROR processing {filename}: {e}")
            skipped.append((filename, str(e)))
            continue

    print(f"\n{'='*70}\nProcessed {len(all_results)}/{len(json_files)} repo(s) successfully "
          f"({len(skipped)} skipped)\n{'='*70}")
    for r in all_results:
        print(f"  {r['source_file']}: exist_f1={r['test_edge_existence_f1']:.4f} "
              f"type_f1={r['test_edge_type_macro_f1']:.4f}")

    if skipped:
        print(f"\nSkipped files:")
        for fname, reason in skipped:
            print(f"  {fname}: {reason}")

    # Persist the aggregate results so you don't have to re-run the whole
    # batch just to look at the numbers again.
    with open("repo_outputs_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()