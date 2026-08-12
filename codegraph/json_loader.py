"""
json_loader.py
--------------
Runs all AST/tree-sitter analysers on source files and writes ONE JSON
output file PER REPO, instead of a single shared output.json.

Why: with a single shared OUTPUT_PATH, every repo processed in a run
(and every subfolder in --master-folder mode) fully overwrote whatever
the previous repo had just written -- so a multi-repo main.py run ended
up with only the LAST repo's data on disk, silently. Each repo now gets
its own isolated output file at:

    repo_outputs/<sanitized-repo-name>.json

e.g. repo_outputs/metaflow.json, repo_outputs/tensorflow.json. Nothing
is merged across repos any more -- each file is a clean, self-contained
snapshot for exactly one repo, which is also what graph_builder.py /
main.py want (one set of graphs per repo).

Two ways to use this module:

1. Programmatically (used by main.py to process an entire repo):
       from json_loader import analyse_all_files
       results = analyse_all_files(["<repo_url>"])
       # results == {repo_url: {"path": "repo_outputs/<name>.json", "output": {...}}}
       # (one entry per distinct repo_url found in the loaded sources --
       #  more than one for --master-folder, which can cover many repos
       #  in a single load_sources() call)

2. Interactively, for re-analysing a single file after an edit:
       python json_loader.py
       > Enter the file path to analyse: src/foo.py

Non-Python, non-tree-sitter-supported files are recorded in file_index
but skipped for AST/tree-sitter analysis.
Every JSON record is tagged with source_file (or filepath), language,
and repo_url.

--- Tree-sitter integration note ---
treesitter_analyzers.extract_repo.extract_repo(repo_dir, repo_url) walks
an ENTIRE repo checkout on disk and returns file_index/class_diagram/
function_call_flow/variables_per_function/data_flow for every
tree-sitter-supported file in that repo in one call. It is NOT a
per-file function (unlike the old assumption this module used to make),
so it's invoked once per repo_url here, not once per file. It also has
no `import_flow` output -- that's a Python-AST-only concept -- so
non-Python repos simply won't have import_flow records.

ASSUMPTION: to call extract_repo() we need the on-disk clone directory
for each repo (not just the repo_url). This module looks for that path
on each source dict under one of a few likely key names (see
_repo_dir_for below). If source_loader.py exposes it under a different
name, update _repo_dir_for accordingly.
"""

import ast
import json
import os
import re
from pathlib import Path

from source_loader import load_sources, cleanup_temp_repos, DEFAULT_CACHE_DIR
from Flow_analyser.class_flow import ClassDiagramAnalyzer
from Flow_analyser.data_flow import DataFlowAnalyzer
from Flow_analyser.function_flow import CallFlowAnalyzer
from Flow_analyser._common import iter_qualified_functions
from Flow_analyser.imports import get_import_aliases, find_usages_in_function
from treesitter_analyzers.extract_repo import extract_repo
from treesitter_analyzers.common import EXTENSION_LANGUAGE_MAP

OUTPUT_PATH = os.path.join("data", "output.json")


def repo_slug(repo_url: str) -> str:
    """Turn a repo URL / local path / master-folder subdir name into a
    filesystem-safe slug, e.g. 'https://github.com/Netflix/metaflow.git'
    -> 'metaflow'. Falls back to 'repo' if nothing usable is left."""
    if not repo_url:
        return "repo"
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return name or "repo"


def output_path_for(repo_url: str) -> str:
    """Return the single shared output JSON path."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    return OUTPUT_PATH

_REQUIRED_KEYS = (
    "file_index",
    "function_call_flow",
    "data_flow",
    "variables_per_function",
    "class_diagram",
    "import_flow",
)

# Candidate keys a source dict (as returned by load_sources()) might use
# to expose the on-disk clone directory for its repo. Tried in order.
_REPO_DIR_KEYS = ("repo_dir", "repo_path", "local_path", "clone_path", "cache_dir")
# NOTE: this used to hardcode "/Users/ankushpal/Desktop/repo-cache" (hyphen),
# which never matched source_loader.py's actual DEFAULT_CACHE_DIR ("repo_cache",
# underscore) -- so this fallback silently never fired for cached repos. Now
# reusing the real constant so the two modules can't drift apart again. In
# practice this fallback is now only a safety net: source_loader.load_sources()
# stamps "repo_dir" directly onto every source dict (local, cached,
# master-folder, and freshly-cloned), so the loop below should find it there
# first.
REPO_ROOT = DEFAULT_CACHE_DIR

# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_output() -> dict:
    return {key: [] for key in _REQUIRED_KEYS}


def _repo_dir_for(src: dict) -> str | None:
    for key in _REPO_DIR_KEYS:
        if src.get(key):
            return src[key]

    repo_url = src.get("repo_url")

    if repo_url:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo_name = repo_name.removesuffix(".git")

        candidate = Path(REPO_ROOT) / repo_name

        if candidate.exists():
            return str(candidate)

    return None

def load_output(path: str = OUTPUT_PATH) -> dict:
    """Load existing output.json, or create an empty structure."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if os.path.exists(path):
        print(f"  ℹ  Existing {path} found — loading existing data.")

        try:
            with open(path, "r", encoding="utf-8") as f:
                output = json.load(f)

            # Make sure all required keys exist
            for key in _REQUIRED_KEYS:
                output.setdefault(key, [])

            return output

        except json.JSONDecodeError:
            print(f"  ⚠  {path} is invalid JSON — starting fresh.")

    return _empty_output()


def drop_file_records(output: dict, repo_url: str, filepath: str) -> None:
    """Remove all existing records that belong to *filepath* WITHIN *repo_url*.

    Keyed on (repo_url, filepath), not filepath alone -- two different
    repos very commonly share the same relative path (setup.py,
    src/__init__.py, ...), and matching on filepath only would silently
    delete one repo's records when analysing another repo that happens
    to have a file at the same path.
    """
    for key in ("file_index", "function_call_flow", "data_flow",
                "variables_per_function", "class_diagram", "import_flow"):
        output[key] = [
            rec for rec in output[key]
            if not (
                rec.get("repo_url") == repo_url
                and (rec.get("filepath") == filepath or rec.get("source_file") == filepath)
            )
        ]


def save_output(output: dict, path: str = OUTPUT_PATH) -> None:
    with open(path, "w") as f:
        json.dump(output, f, indent=2)

    analysed = sum(1 for fi in output["file_index"] if fi["analysed"])
    total    = len(output["file_index"])
    print()
    print("=" * 60)
    print(f"JSON synced  → {path}")
    print(f"  Total files tracked : {total}")
    print(f"  Analysed            : {analysed}")
    print(f"  Skipped             : {total - analysed}")
    print("=" * 60)


def analyse_treesitter_repo(output: dict, repo_url: str, repo_dir: str) -> None:
    """
    Run the tree-sitter analysers over an ENTIRE repo checkout in one
    shot -- extract_repo(repo_dir, repo_url) walks repo_dir itself, it
    is not a per-file function -- then merge the result into *output*
    the same way analyse_file() merges a single Python file's records.

    Does NOT save to disk -- call save_output() once after processing
    everything you want in this batch.
    """
    print(f"  Tree-sitter pass over repo: {repo_url}")
    ts_result = extract_repo(repo_dir, repo_url)

    # Drop any previously-stored records for every file this pass
    # covers, exactly like analyse_file() does for a single file --
    # otherwise a re-run would append duplicate records on top of stale
    # ones instead of replacing them.
    for fi in ts_result["file_index"]:
        drop_file_records(output, repo_url, fi["filepath"])

    output["file_index"].extend(ts_result["file_index"])
    output["class_diagram"].extend(ts_result["class_diagram"])
    output["function_call_flow"].extend(ts_result["function_call_flow"])
    output["variables_per_function"].extend(ts_result["variables_per_function"])
    output["data_flow"].extend(ts_result["data_flow"])
    # NOTE: extract_repo() has no import_flow output -- that's a
    # Python-AST-only concept (Flow_analyser.imports). Non-Python repos
    # simply won't get import_flow records, same as before this change.

    n = len(ts_result["file_index"])
    print(f"    ✓ {n} files merged from tree-sitter pass")


def analyse_file(output: dict, src: dict) -> None:
    """
    Analyse a single Python source dict (as returned by load_sources())
    via the Python AST analysers, and merge its records into *output*
    in-place. Does NOT save to disk — call save_output() once after
    processing all files you want in this batch.

    Non-Python files are only recorded in file_index here as
    unsupported/skipped: tree-sitter-supported languages are handled
    up front, once per repo, by analyse_treesitter_repo() instead (see
    analyse_all_files()) -- extract_repo() cannot be called per-file.
    """
    filepath = src["filepath"]
    filename = src["filename"]
    language = src["language"]
    repo_url = src.get("repo_url", "")
    code     = src["code"]

    drop_file_records(output, repo_url, filepath)

    print(f"  Analysing  {filepath}  [{language}]")

    if language != "Python":
        # Tree-sitter-supported files are handled at the repo level in
        # analyse_all_files() and never reach this branch for a normal
        # run; if one does (e.g. interactive single-file mode with no
        # repo_dir available), it's simply recorded as skipped here.
        output["file_index"].append({
            "filepath": filepath,
            "filename": filename,
            "language": language,
            "repo_url": repo_url,
            "analysed": False,
            "reason": f"Unsupported language ({language})",
        })
        return

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        print(f"    ✗  SyntaxError: {exc}")
        output["file_index"].append({
            "filepath": filepath,
            "filename": filename,
            "language": language,
            "repo_url": repo_url,
            "analysed": False,
            "reason":   f"SyntaxError: {exc}",
        })
        return

    call_analyzer  = CallFlowAnalyzer();   call_analyzer.visit(tree)
    data_analyzer  = DataFlowAnalyzer();   data_analyzer.visit(tree)
    class_analyzer = ClassDiagramAnalyzer(); class_analyzer.visit(tree)

    tag = {"source_file": filepath, "language": language, "repo_url": repo_url}

    call_flow_map: dict[str, list[str]] = {}
    for caller, callee in call_analyzer.calls:
        call_flow_map.setdefault(caller, [])
        if callee not in call_flow_map[caller]:
            call_flow_map[caller].append(callee)

    for fn, callees in call_flow_map.items():
        output["function_call_flow"].append({"function": fn, "calls": callees, **tag})

    for fn, var, callee, pos in data_analyzer.data_flows:
        output["data_flow"].append({
            "function": fn, "variable": var,
            "passed_to": callee, "arg_position": pos, **tag,
        })

    for fn, vars_ in data_analyzer.assignments.items():
        output["variables_per_function"].append({
            "function": fn,
            "params":   [v[6:] for v in vars_ if v.startswith("param:")],
            "assigns":  [v      for v in vars_ if not v.startswith("param:")],
            **tag,
        })

    for cls, info in class_analyzer.classes.items():
        output["class_diagram"].append({
            "class": cls, "bases": info["bases"],
            "attributes": info["attributes"], "methods": info["methods"],
            **tag,
        })

    # ── Import detail: which imports each function actually uses ─────────
    aliases = get_import_aliases(tree)
    for qualified_fn, func_node in iter_qualified_functions(tree):
        used_modules = find_usages_in_function(func_node, aliases)
        if used_modules:
            output["import_flow"].append({
                "function": qualified_fn,
                "imports_used": sorted(used_modules),
                **tag,
            })

    output["file_index"].append({
        "filepath": filepath,
        "filename": filename,
        "language": language,
        "repo_url": repo_url,
        "analysed": True,
    })

    print(f"    ✓  Done")


def analyse_all_files(args: list[str] | None = None) -> dict:
    """
    Load repo(s) (via source_loader.load_sources(args)), analyse EVERY
    discovered file, and write ONE JSON OUTPUT FILE PER REPO. This is
    what main.py calls to process each job non-interactively.

    A single load_sources(args) call can return files belonging to more
    than one repo_url (e.g. --master-folder covers every subdirectory
    in one call), so this function groups sources by repo_url FIRST and
    then runs the two analysis passes independently, per repo, so that
    one repo's records can never leak into or overwrite another repo's
    output file:

      1. Tree-sitter pass, once per repo_url -- extract_repo() walks
         the whole repo checkout itself and covers every JS/TS/Go/
         Java/C/C++ file in one call.
      2. Python AST pass, once per Python file in that repo (files
         already covered by pass 1 are skipped so they don't get
         double-processed or clobbered back to analysed=False).

    Returns a dict keyed by repo_url:
        {
          "<repo_url>": {"path": "repo_outputs/<slug>.json", "output": {...}},
          ...
        }

    *args* defaults to sys.argv[1:] (single-repo CLI usage, unchanged
    from before). Pass an explicit ["<url>", "--branch", "<name>"] list
    to process a specific repo -- this is what lets main.py loop over
    several jobs in one run while keeping every repo's output separate.
    """
    print("=" * 60)
    print("LOADING REPOSITORY FILES")
    print("=" * 60)

    sources = load_sources(args)

    # Group sources by repo_url so each repo is analysed in complete
    # isolation from every other repo in this call (this is what fixes
    # the old bug where a multi-repo run silently kept only the last
    # repo's data, since everything used to share one output.json).
    sources_by_repo: dict[str, list[dict]] = {}
    for src in sources:
        sources_by_repo.setdefault(src.get("repo_url", ""), []).append(src)

    results: dict[str, dict] = {}

    for repo_url, repo_sources in sources_by_repo.items():
        print("\n" + "-" * 60)
        print(f"ANALYSING REPO: {repo_url or '(unknown repo_url)'}")
        print("-" * 60)

        out_path = output_path_for(repo_url)
        output = load_output(out_path)

        # --- Pass 1: tree-sitter, once for this repo ------------------
        repo_dir = None
        for src in repo_sources:
            repo_dir = _repo_dir_for(src)
            if repo_dir:
                break
        if repo_dir:
            analyse_treesitter_repo(output, repo_url, repo_dir)
        else:
            print(
                f"  ⚠  No on-disk repo_dir found for {repo_url!r} on its source "
                f"dicts -- tree-sitter-supported files in this repo will be "
                f"recorded as skipped instead of analysed. See _repo_dir_for()."
            )

        ts_handled: set[tuple[str, str]] = {
            (fi.get("repo_url", ""), fi["filepath"])
            for fi in output["file_index"]
        }

        # --- Pass 2: Python AST, per file, skipping anything pass 1 covered
        for src in repo_sources:
            key = (src.get("repo_url", ""), src["filepath"])
            ext = os.path.splitext(src["filepath"])[1].lower()

            if src["language"] != "Python" and ext in EXTENSION_LANGUAGE_MAP and key in ts_handled:
                # Already analysed (or recorded as attempted) by the
                # tree-sitter repo-level pass above -- don't reprocess.
                continue

            analyse_file(output, src)

        save_output(output, out_path)
        results[repo_url] = {"path": out_path, "output": output}

    # Now that every repo_dir above has actually been walked by
    # extract_repo(), it's safe to delete any fresh clones that
    # source_loader.py deferred cleanup on. Cached/local/master-folder
    # repos were never registered there, so they're untouched.
    cleanup_temp_repos()

    return results


# ── Interactive single-file mode (only runs when this file is executed
#    directly, e.g. `python json_loader.py`, for re-syncing one file after
#    an edit) ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SINGLE-FILE AST ANALYSER")
    print("=" * 60)
    target_filepath = input("\nEnter the file path to analyse: ").strip()

    if not target_filepath:
        raise SystemExit("No file path provided — exiting.")

    print()
    print("=" * 60)
    print("LOADING REPOSITORY FILES")
    print("=" * 60)

    sources = load_sources()
    src_map: dict[str, dict] = {s["filepath"]: s for s in sources}

    if target_filepath not in src_map:
        available = "\n  ".join(src_map.keys()) or "(none)"
        raise SystemExit(
            f"\n✗  '{target_filepath}' not found in the repository.\n"
            f"Available paths:\n  {available}"
        )

    target_src = src_map[target_filepath]
    repo_url = target_src.get("repo_url", "")
    out_path = output_path_for(repo_url) if repo_url else OUTPUT_PATH
    output = load_output(out_path)
    ext = os.path.splitext(target_filepath)[1].lower()

    if target_src["language"] != "Python" and ext in EXTENSION_LANGUAGE_MAP:
        repo_dir = _repo_dir_for(target_src)
        if repo_dir:
            # extract_repo() can't target a single file -- it re-walks
            # the whole repo, but that's still cheaper/correct here.
            analyse_treesitter_repo(output, repo_url, repo_dir)
        else:
            print(
                "  ⚠  No repo_dir available for this file's repo -- falling "
                "back to recording it as skipped."
            )
            analyse_file(output, target_src)
    else:
        analyse_file(output, target_src)

    save_output(output, out_path)
    cleanup_temp_repos()