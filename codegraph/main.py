"""
main.py
-------
Entry point.

Usage
-----
    python main.py https://github.com/owner/repo
    python main.py https://github.com/owner/repo --branch develop
    python main.py https://github.com/ownerA/repoA https://github.com/ownerB/repoB
    python main.py --repos-file repos.txt
    python main.py --repos-file repos.txt https://github.com/owner/extraRepo
    python main.py --master-folder /path/to/master_folder
    python main.py --master-folder ~/Desktop/repo_cache https://github.com/owner/extraRepo

    python main.py --master-folder /Users/ankushpal/Desktop/repo_cache

    python main.py https://github.com/owner/repo --cache-dir ~/Desktop/repo_cache

--master-folder points at a local directory that itself contains one
subdirectory per already-cloned repo (no git access needed -- see
source_loader.py's master-folder mode). It can be combined with
inline URLs and/or --repos-file in the same run.

--cache-dir works the other way around: you still give a GitHub link,
but before cloning it, source_loader first checks whether a folder
named after that repo already exists inside --cache-dir. If it does,
that local copy is used directly and the clone is skipped; if not, the
repo is cloned from GitHub as normal. Defaults to
/Users/ankushpal/Desktop/repo_cache if not given.

Every repo passed (inline, --repos-file, and/or --master-folder) is
analysed into its OWN JSON file under repo_outputs/<repo-name>.json
(see json_loader.py) -- repos are no longer merged together, so one
repo's data can never silently overwrite another's. Graphs are then
built per repo and exported as interactive HTML under
graphs/<repo-name>/.
"""

import os
import sys

from json_loader import analyse_all_files, repo_slug   # analyses EVERY file, writes repo_outputs/<repo>.json
from source_loader import DEFAULT_CACHE_DIR

#------------------------------------file Naming-----------------------------------





# ── Path normalization ──────────────────────────────────────────────────────

def _normalize_path(path: str) -> str:
    """Expand '~' and strip stray whitespace, so paths like
    '~/Desktop/repo_cache' or '../repo_cache ' (trailing space) work
    regardless of how they were typed or passed through."""
    return os.path.expanduser(path.strip())


# ── Multi-repo argument parsing ────────────────────────────────────────────────

def _split_repo_args(argv):
    """Split a list of tokens into one or more (url, branch) jobs.

    Supports:
        <url1> <url2> <url3>
        <url1> --branch dev <url2> --branch main
    Each URL may be optionally followed by --branch <name> applying
    only to that URL.
    """
    jobs = []
    i = 0
    current_url = None
    current_branch = None

    def flush():
        if current_url is not None:
            jobs.append((current_url, current_branch))

    while i < len(argv):
        tok = argv[i]
        if tok in ("--branch", "-b") and i + 1 < len(argv):
            current_branch = argv[i + 1]
            i += 2
        else:
            flush()
            current_url = tok
            current_branch = None
            i += 1
    flush()
    return jobs


def _extract_repo_file_flags(argv):
    """Pull any --repos-file/-f <path> flags out of argv.

    Returns (remaining_argv, list_of_paths) so inline URLs/--branch
    flags can still be combined with one or more --repos-file flags
    in the same run.
    """
    remaining = []
    paths = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--repos-file", "-f") and i + 1 < len(argv):
            paths.append(_normalize_path(argv[i + 1]))
            i += 2
        else:
            remaining.append(tok)
            i += 1
    return remaining, paths


def _extract_cache_dir_flag(argv):
    """Pull a --cache-dir/-c <path> flag out of argv (global for the
    run). Returns (remaining_argv, cache_dir_or_None).

    When set, each cloned-repo job first checks this folder for an
    already-available copy (named after the repo) before doing a
    `git clone` -- see source_loader.py's cache-dir check."""
    remaining = []
    cache_dir = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--cache-dir", "-c") and i + 1 < len(argv):
            cache_dir = _normalize_path(argv[i + 1])
            i += 2
        else:
            remaining.append(tok)
            i += 1
    return remaining, cache_dir


def _extract_master_folder_flags(argv):
    """Pull any --master-folder/-m <path> flags out of argv.

    Returns (remaining_argv, list_of_paths). Supports multiple
    --master-folder flags in one run, same pattern as --repos-file.
    """
    remaining = []
    paths = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--master-folder", "-m") and i + 1 < len(argv):
            paths.append(_normalize_path(argv[i + 1]))
            i += 2
        else:
            remaining.append(tok)
            i += 1
    return remaining, paths


def _load_repo_file(path):
    """Parse a text file of repo links into (url, branch) jobs.

    Format: one repo per line, optionally followed by a branch name.
        https://github.com/org/repoA
        https://github.com/org/repoB  develop
        # lines starting with # and blank lines are ignored
    """
    jobs = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            url = parts[0]
            branch = parts[1] if len(parts) > 1 else None
            if len(parts) > 2:
                print(f"  ⚠  {path}:{lineno}: ignoring extra tokens {parts[2:]!r}")
            jobs.append((url, branch))
    return jobs


def _collect_jobs(argv):
    """Combine repo links from --repos-file <path> flag(s) and/or inline
    URL args into a single ordered list of (url, branch) jobs.

    Local paths given inline (not via --master-folder) are also
    accepted here and passed straight through as the 'url' slot --
    source_loader.load_sources() knows how to treat a local directory
    as a single already-available repo."""
    remaining_argv, repo_files = _extract_repo_file_flags(argv)

    jobs = []
    for path in repo_files:
        try:
            file_jobs = _load_repo_file(path)
        except OSError as e:
            print(f"Could not read repos file {path!r}: {e}")
            continue
        print(f"Loaded {len(file_jobs)} repo link(s) from {path!r}")
        jobs.extend(file_jobs)

    jobs.extend(_split_repo_args(remaining_argv))
    return jobs


# ── Graph building + export ─────────────────────────────────────────────────

'''GRAPH_OUTPUT_DIR = "graphs"


def _build_and_export_graphs(data: dict, out_dir: str) -> None:
    """Build all four graphs from one repo's data and write each out as
    an interactive pyvis HTML file into *out_dir* (one subfolder per
    repo, e.g. graphs/metaflow/call_graph.html)."""
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        ("Call Graph",           build_call_graph,          "call_graph.html"),
        ("Dependency Graph",     build_dependency_graph,    "dependency_graph.html"),
        ("Data Flow Graph",      build_data_flow_graph,     "data_flow_graph.html"),
        ("Class Diagram",        build_class_diagram_graph, "class_diagram.html"),
    ]

    for title, builder_fn, filename in specs:
        G = builder_fn(data)
        out_path = os.path.join(out_dir, filename)
        build_pyvis(G, title=title, filename=out_path)'''


# ── Run the pipeline over one or more repos (non-interactive) ─────────────────
raw_argv = sys.argv[1:]
raw_argv, master_folders = _extract_master_folder_flags(raw_argv)
raw_argv, cache_dir = _extract_cache_dir_flag(raw_argv)
jobs = _collect_jobs(raw_argv)

if not jobs and not master_folders:
    print(
        "Usage:\n"
        "  python main.py <repo_url> [<repo_url> ...]\n"
        "  python main.py <repo_url> --branch <branch_name>\n"
        "  python main.py --repos-file <path> [<repo_url> ...]\n"
        "  python main.py --master-folder <path>\n"
        "  python main.py <repo_url> --cache-dir <path>\n\n"
        "--repos-file points to a text file with one repo link per line\n"
        "(optionally followed by a branch name).\n\n"
        "--master-folder points to a local directory containing one\n"
        "subdirectory per already-cloned repo -- no git access needed.\n\n"
        "--cache-dir points to a local directory laid out the same way\n"
        "as --master-folder (one subdirectory per repo, named after the\n"
        "repo). For each <repo_url> given, this folder is checked first --\n"
        "if a matching subdirectory is found it's used directly and the\n"
        "git clone is skipped entirely; otherwise the repo is cloned from\n"
        "GitHub as usual. Defaults to "
        + repr(DEFAULT_CACHE_DIR)
        + " if not given."
    )
    raise SystemExit(1)

succeeded, failed = [], []
all_repo_results: dict[str, dict] = {}   # repo_url -> {"path": ..., "output": {...}}

for folder in master_folders:
    print(f"\n=== Analysing master folder {folder} ===")
    try:
        res = analyse_all_files(["--master-folder", folder])
        all_repo_results.update(res)
        succeeded.append(f"master-folder:{folder}")
    except Exception as e:
        print(f"Error processing master folder {folder!r}: {e}")
        failed.append((folder, str(e)))

for url, branch in jobs:
    print("\n" + "=" * 80)
    print(f"STARTING REPOSITORY: {url}")
    print(f"BRANCH: {branch or 'default'}")
    print("=" * 80)

    repo_args = [url]

    if branch:
        repo_args += ["--branch", branch]

    if cache_dir:
        repo_args += ["--cache-dir", cache_dir]

    print("[1/3] Starting analyse_all_files()")
    print("Arguments:", repo_args)
    try:
        res = analyse_all_files(repo_args)

        print("[2/3] analyse_all_files() finished.")

        all_repo_results.update(res)
        succeeded.append(url)

        print(f"[3/3] Repository completed: {url}")

    except Exception as e:
        print(f"[ERROR] {url}")
        print(e)
        failed.append((url, str(e)))

if failed:
    print(f"\n── {len(failed)}/{len(jobs) + len(master_folders)} repo(s) failed:")
    for url, err in failed:
        print(f"  ✗ {url}\n      {err}")

if not succeeded:
    print(
        "\nNo repos were successfully analysed in this run -- "
        f"output.json was not created/updated, so there's nothing to "
        "build graphs from. Check the errors above (common causes: "
        "repo doesn't exist / was renamed, no network access, git "
        "isn't configured for this host, or a --master-folder path "
        "that doesn't exist / has no repo subdirectories)."
    )
    raise SystemExit(1)

# ── Print file index summary, grouped by repo (each repo has its own
#    JSON file now -- see json_loader.output_path_for()) ──────────────────
total_files = sum(len(info["output"]["file_index"]) for info in all_repo_results.values())
print(f"\n── Repositories analysed: {len(all_repo_results)}")
print(f"── Total files found    : {total_files}")
for repo_url, info in all_repo_results.items():
    files = info["output"]["file_index"]
    print(f"\n  {repo_url}  ({len(files)} files)  →  {info['path']}")
    for fi in files:
        status = "✓" if fi["analysed"] else "✗"
        note   = f"  — {fi.get('reason','')}" if not fi["analysed"] else ""
        print(f"    [{status}] {fi['filepath']}  ({fi['language']}){note}")

# ── Build graphs, once per repo, into graphs/<repo-name>/ ──────────────────
print("\n" + "=" * 60)
print("BUILDING GRAPHS (one set per repo)")
print("=" * 60)
