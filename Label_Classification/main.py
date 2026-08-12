"""
main.py
-------
Connects to EXISTING code:
  - source_loader.py's load_sources() -- used to fetch each repo's files
    (from --cache-dir if already cloned, else a fresh `git clone`).

Produces NEW files:
  - one Prediction_<repo-name>.json per repo, written to --out-dir
    (default ~/work/predictions), in the schema merge_predictions.py and
    infer_codebert.py both already expect:
        [{"filepath", "repo_url", "predicted_label", "confidence",
          "is_confident", "all_labels"}, ...]

This is the HEURISTIC (rule-based) classifier -- no model, no internet
required. It exists to bootstrap weak-supervision labels for every file
in run.py's ~150 repos so build_dataset.py has something to train
train_codebert.py on. infer_codebert.py later re-classifies the same
repos with the fine-tuned model and writes output in the identical
schema, so the two are directly comparable.

Labels
------
  test                  test/spec files
  documentation         markdown/rst/plaintext docs
  configuration          config/manifest/env files
  build_tooling         build scripts, CI, packaging
  example                sample/demo/tutorial code
  vendored_dependency    third-party code vendored into the repo
  entry_point            an app/service's main entry file
  utility                small helper/util modules
  core_logic             everything else -- the actual application logic

Usage
-----
    python main.py https://github.com/owner/repo1 https://github.com/owner/repo2
    python main.py https://github.com/owner/repo --branch develop
    python main.py https://github.com/owner/repo --cache-dir ~/repo_cache
    python main.py --master-folder /path/to/master_folder
    python main.py https://github.com/owner/repo --out-dir /custom/predictions
"""
import argparse
import json
import os
import re

from source_loader import load_sources, cleanup_temp_repos, _repo_name_from_url

DEFAULT_OUT_DIR = os.path.expanduser("~/work/predictions")
CONFIDENCE_THRESHOLD = 0.5   # matches infer_codebert.py's is_confident cutoff
MAX_CHARS = 4000             # cap how much of a file's content we scan
LABELS = [
    "test", "documentation", "configuration", "build_tooling", "example",
    "vendored_dependency", "entry_point", "utility", "core_logic",
]

# ── Path-based rules ─────────────────────────────────────────────────────
# Each rule: (compiled regex against the lowercased repo-relative path,
# label, point value). Path rules dominate the score since a file's
# location is usually the strongest signal for what role it plays.
_PATH_RULES = [
    (re.compile(r"(^|/)(tests?|specs?|__tests__)(/|$)"), "test", 6),
    (re.compile(r"(^|/)(test_|_test\.|\.test\.|\.spec\.)"), "test", 5),
    (re.compile(r"\.(md|rst|txt|adoc)$"), "documentation", 5),
    (re.compile(r"(^|/)docs?(/|$)"), "documentation", 4),
    (re.compile(r"readme|changelog|contributing|license", re.I), "documentation", 4),
    (re.compile(r"\.(ya?ml|toml|ini|cfg|conf|env)$"), "configuration", 5),
    (re.compile(r"(^|/)(package|composer|pyproject|tsconfig|babel\.config|\.eslintrc)"), "configuration", 5),
    (re.compile(r"dockerfile|docker-compose", re.I), "configuration", 4),
    (re.compile(r"(^|/)(scripts?|tools?|ci|\.circleci|\.github)(/|$)"), "build_tooling", 4),
    (re.compile(r"(^|/)(makefile|setup\.py|webpack\.config|rollup\.config|gulpfile|gruntfile)", re.I), "build_tooling", 5),
    (re.compile(r"(^|/)(examples?|demos?|samples?|tutorials?)(/|$)"), "example", 5),
    (re.compile(r"(^|/)(vendor|third[-_]?party|external|deps)(/|$)"), "vendored_dependency", 6),
    (re.compile(r"(^|/)(main|index|app|server|__main__|cmd)\.(py|js|ts|go|java|rb)$"), "entry_point", 4),
    (re.compile(r"(^|/)(utils?|helpers?|common|shared)(/|\.py$|\.js$|\.ts$)"), "utility", 4),
]


def _score_path(rel_path):
    scores = {label: 0.0 for label in LABELS}
    p = rel_path.lower()
    for pattern, label, points in _PATH_RULES:
        if pattern.search(p):
            scores[label] += points
    return scores


def _score_content(code):
    """Cheap structural signals from the file body itself -- used mainly
    to separate entry_point/utility/core_logic when the path gives no
    strong hint."""
    scores = {label: 0.0 for label in LABELS}
    snippet = code[:MAX_CHARS]

    n_lines = snippet.count("\n") + 1
    n_funcs = len(re.findall(r"\bdef \w+\(|\bfunction \w*\(|\b\w+\s*=\s*\([^)]*\)\s*=>", snippet))
    n_classes = len(re.findall(r"\bclass \w+", snippet))
    has_main_guard = bool(re.search(r'if __name__ == ["\']__main__["\']', snippet))
    has_cli_parse = bool(re.search(r"argparse|yargs|commander|click\.command", snippet))
    comment_lines = len(re.findall(r"^\s*(#|//)", snippet, re.M))

    if has_main_guard or has_cli_parse:
        scores["entry_point"] += 3
    if n_classes >= 2:
        scores["core_logic"] += 2
    if n_funcs >= 4 and n_classes == 0:
        scores["utility"] += 2
    if n_lines > 0 and comment_lines / n_lines > 0.35:
        scores["documentation"] += 1
    if n_lines < 15:
        # tiny files are rarely the "real" logic of a repo
        scores["utility"] += 1

    return scores


def classify_file(rel_path, code):
    path_scores = _score_path(rel_path)
    content_scores = _score_content(code)

    combined = {label: path_scores[label] + content_scores[label] for label in LABELS}

    # Baseline so every file has *some* mass on core_logic -- the
    # fallback label when nothing else fires.
    combined["core_logic"] += 1.5

    total = sum(combined.values())
    ranked = sorted(
        [(label, combined[label] / total) for label in LABELS],
        key=lambda x: x[1], reverse=True,
    )
    top_label, top_conf = ranked[0]
    is_confident = top_conf >= CONFIDENCE_THRESHOLD
    all_labels = [
        {"label": l, "confidence": round(c, 4)} for l, c in ranked if c >= CONFIDENCE_THRESHOLD
    ]

    return top_label, round(top_conf, 4), is_confident, (all_labels if is_confident else [])


def classify_repo(sources, repo_url):
    results = []
    for s in sources:
        content = s["code"][:MAX_CHARS]
        label, conf, is_confident, all_labels = classify_file(s["filepath"], content)
        results.append({
            "filepath": s["filepath"],
            "repo_url": repo_url,
            "predicted_label": label,
            "confidence": conf,
            "is_confident": is_confident,
            "all_labels": all_labels,
        })
    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Heuristically classify every source file in one or more repos."
    )
    ap.add_argument("repos", nargs="*", help="repo URL(s) or local path(s)")
    ap.add_argument("--branch", "-b", default=None)
    ap.add_argument("--cache-dir", "-c", default=None,
                     help="persistent repo cache -- forwarded to source_loader")
    ap.add_argument("--master-folder", "-m", default=None,
                     help="local folder containing one subdirectory per already-cloned repo")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                     help="where to write Prediction_<repo>.json files (default: %(default)s)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.master_folder:
        sources = load_sources(["--master-folder", args.master_folder])
        by_repo = {}
        for s in sources:
            by_repo.setdefault(s["repo_url"], []).append(s)
        for repo_url, repo_sources in by_repo.items():
            results = classify_repo(repo_sources, repo_url)
            out_path = os.path.join(args.out_dir, f"Prediction_{repo_url}.json")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  {repo_url}: {len(results)} file(s) classified -> {out_path}")
        return

    if not args.repos:
        raise SystemExit(
            "No repository given.\n"
            "Usage:  python main.py https://github.com/owner/repo [https://github.com/owner/repo2 ...]\n"
            "        python main.py --master-folder /path/to/master_folder"
        )

    for repo_url in args.repos:
        single_args = [repo_url]
        if args.branch:
            single_args += ["--branch", args.branch]
        if args.cache_dir:
            single_args += ["--cache-dir", args.cache_dir]

        print(f"\n=== {repo_url} ===")
        try:
            sources = load_sources(single_args)
        except Exception as exc:
            print(f"  ✗ failed to load {repo_url!r}: {exc}")
            continue

        results = classify_repo(sources, repo_url)

        name = _repo_name_from_url(repo_url) if repo_url.startswith(("http://", "https://")) \
            else os.path.basename(os.path.normpath(repo_url))
        out_path = os.path.join(args.out_dir, f"Prediction_{name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  {len(results)} file(s) classified -> {out_path}")

        # No-op when --cache-dir was given (load_sources leaves cached/
        # persisted repos on disk and never registers them for cleanup);
        # only cleans up genuinely temporary clones (no cache dir).
        cleanup_temp_repos()


if __name__ == "__main__":
    main()