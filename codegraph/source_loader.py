"""
source_loader.py
----------------
Loads source files from a GitHub/GitLab repo URL, clones it to a temp
directory, walks all supported-language files, then deletes the clone.

Before cloning, a repo URL is first checked against a local cache
directory (--cache-dir, default DEFAULT_CACHE_DIR below). If a
subdirectory named after the repo already exists there, that copy is
used directly and the `git clone` is skipped.

Usage
-----
    python main.py https://github.com/owner/repo
    python main.py https://github.com/owner/repo --branch develop
    python main.py https://github.com/owner/repo --cache-dir ~/Desktop/repo_cache
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

# ── Language detection ────────────────────────────────────────────────────────
_EXT_TO_LANG: dict[str, str] = {
    ".py":   "Python",
    ".js":   "JavaScript",
    ".ts":   "TypeScript",
    ".jsx":  "JavaScript (JSX)",
    ".tsx":  "TypeScript (TSX)",
    ".java": "Java",
    ".cpp":  "C++",
    ".cc":   "C++",
    ".cxx":  "C++",
    ".c":    "C",
    ".h":    "C/C++ Header",
    ".cs":   "C#",
    ".go":   "Go",
    ".rb":   "Ruby",
    ".rs":   "Rust",
    ".kt":   "Kotlin",
    ".swift":"Swift",
    ".php":  "PHP",
    ".R":    "R",
    ".scala":"Scala",
    ".sh":   "Shell",
    ".bash": "Shell",
    ".lua":  "Lua",
    ".m":    "MATLAB / Objective-C",
    ".pl":   "Perl",
}

# Directories that are almost never worth analysing
_SKIP_DIRS = {
    ".git", ".github", ".gitlab", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build", ".tox",
    "site-packages", ".mypy_cache", ".pytest_cache", ".eggs",
    "migrations",   # Django auto-generated
}

# Default local cache folder to check for an already-cloned copy of a
# repo before falling back to `git clone`. Layout is the same as
# --master-folder: one subdirectory per repo, named after the repo.
# Override per-run with --cache-dir/-c <path>.
DEFAULT_CACHE_DIR = os.path.expanduser("/Users/ankushpal/Desktop/repo_cache")

# Temp clone directories created by load_sources() for repos that had to
# be freshly `git clone`'d (i.e. NOT already-cached/local/master-folder
# repos). These are deliberately NOT deleted the moment load_sources()
# returns anymore -- json_loader.py's tree-sitter pass needs the on-disk
# checkout (via each source dict's "repo_dir" key) *after* load_sources()
# has already returned. Call cleanup_temp_repos() once that pass is done
# with them, and no earlier.
_TEMP_CLONE_DIRS: list[str] = []


def cleanup_temp_repos() -> None:
    """Delete every temp clone directory registered by load_sources()
    since the last call to this function. Safe to call even if nothing
    is pending (no-op). Cached/local/master-folder repos are never
    touched here -- only fresh `git clone`s land in _TEMP_CLONE_DIRS."""
    while _TEMP_CLONE_DIRS:
        d = _TEMP_CLONE_DIRS.pop()
        shutil.rmtree(d, ignore_errors=True)
        print(f"  Deleted clone → {d}")


def detect_language(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    return _EXT_TO_LANG.get(ext, f"Unknown ({ext})" if ext else "Unknown")


def _is_repo_url(s: str) -> bool:
    return bool(re.match(r"https?://", s)) or s.endswith(".git")


def _is_local_dir(s: str) -> bool:
    """True if *s* points at an existing local directory (and isn't a
    remote URL) -- i.e. an already-available repo checkout on disk."""
    return os.path.isdir(s) and not _is_repo_url(s)


def _repo_name_from_url(url: str) -> str:
    """Derive the repo's folder name from a GitHub/GitLab URL, e.g.
    'https://github.com/owner/repo' or '.../repo.git' -> 'repo'."""
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def _find_cached_repo(url: str, cache_dir: str) -> str | None:
    """If a copy of *url* already exists under *cache_dir* (a folder
    named after the repo), return its path -- otherwise None."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    repo_name = _repo_name_from_url(url)
    candidate = os.path.join(cache_dir, repo_name)
    return candidate if os.path.isdir(candidate) else None


def _clone_repo(url: str, branch: str | None = None) -> str:
    """Clone *url* into a fresh temp dir and return the dir path."""
    tmp = tempfile.mkdtemp(prefix="repo_analyser_")
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, tmp]
    print(f"  Cloning {url!r} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed:\n{result.stderr.strip()}"
        )
    print(f"  Cloned  → {tmp}")
    return tmp


def _walk_repo(root: str) -> list[str]:
    """Return every file in *root* whose extension is in _EXT_TO_LANG."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune unwanted dirs in-place so os.walk won't descend into them
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _EXT_TO_LANG:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def _load_single_local_repo(path: str, repo_url: str) -> list[dict]:
    """
    Read source files directly from an already-available local repo
    directory -- no git clone, and the folder is left untouched
    afterwards (unlike the remote-clone path, which deletes the temp
    checkout when done).

    *repo_url* is whatever identifier should be stamped onto each
    record's "repo_url" field (usually the repo's folder name).
    """
    all_files = _walk_repo(path)
    print(f"  Found {len(all_files)} supported source file(s) in local repo {path!r}")

    sources: list[dict] = []
    for fpath in all_files:
        lang = detect_language(fpath)
        rel  = os.path.relpath(fpath, path)
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                code = fh.read()
        except OSError as exc:
            print(f"  ⚠  Could not read {rel!r}: {exc}")
            continue

        sources.append({
            "filepath": rel,
            "filename": os.path.basename(fpath),
            "repo_url": repo_url,
            "repo_dir": path,
            "language": lang,
            "code":     code,
        })
        print(f"  Loaded  {rel!r}  [{lang}]")

    return sources


def load_sources(args: list[str] | None = None) -> list[dict]:
    """
    Clone a Git repo, read every supported source file, delete the clone,
    and return a list of dicts:
        {
            "filepath": str,   # path inside the temp clone (for reference)
            "filename": str,   # basename
            "repo_url": str,   # original URL
            "language": str,   # detected language
            "code":     str,   # full source text
        }

    *args* defaults to sys.argv[1:].
    Expected format:
        <url>  [--branch <name>]
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        raise ValueError(
            "No repository URL given.\n"
            "Usage:  python main.py https://github.com/owner/repo\n"
            "        python main.py https://github.com/owner/repo --branch dev\n"
            "        python main.py --master-folder /path/to/master_folder"
        )

    # ── Master-folder mode ───────────────────────────────────────────────
    # A local directory that itself contains one subdirectory per
    # already-cloned repo, e.g.:
    #   master_folder/
    #     tensorflow/   (a full repo checkout)
    #     pytorch/      (a full repo checkout)
    # Every immediate subdirectory is treated as its own repo (repo_url
    # = subfolder name). No cloning and nothing is deleted.
    for flag in ("--master-folder", "-m"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 >= len(args):
                raise ValueError(f"{flag} requires a path argument")
            master_path = args[idx + 1]

            if not os.path.isdir(master_path):
                raise ValueError(f"Master folder not found: {master_path!r}")

            repo_dirs = sorted(
                d for d in os.listdir(master_path)
                if os.path.isdir(os.path.join(master_path, d)) and d not in _SKIP_DIRS
            )
            if not repo_dirs:
                raise ValueError(f"No repo subdirectories found in {master_path!r}")

            print(f"  Master folder {master_path!r} → {len(repo_dirs)} repo(s)")
            all_sources: list[dict] = []
            for repo_name in repo_dirs:
                repo_path = os.path.join(master_path, repo_name)
                print(f"\n  ── Repo: {repo_name}")
                all_sources.extend(_load_single_local_repo(repo_path, repo_url=repo_name))
            return all_sources

    # Parse URL/local-path and optional --branch / --cache-dir flags
    url       = None
    branch    = None
    cache_dir = DEFAULT_CACHE_DIR
    i = 0
    while i < len(args):
        if args[i] in ("--branch", "-b") and i + 1 < len(args):
            branch = args[i + 1]
            i += 2
        elif args[i] in ("--cache-dir", "-c") and i + 1 < len(args):
            cache_dir = os.path.expanduser(args[i + 1].strip())
            i += 2
        elif _is_repo_url(args[i]) or _is_local_dir(args[i]):
            url = args[i]
            i += 1
        else:
            raise ValueError(f"Unrecognised argument: {args[i]!r}")

    if url is None:
        raise ValueError("No valid repository URL or local path found in arguments.")

    # ── Local single-repo path (already on disk, e.g. inside a master
    # folder you're pointing at directly) — no clone, no cleanup ───────────
    if _is_local_dir(url):
        repo_name = os.path.basename(os.path.normpath(url))
        print(f"  Using local repo {url!r} (repo_url={repo_name!r})")
        return _load_single_local_repo(url, repo_url=repo_name)

    # ── Check cache dir before cloning ───────────────────────────────────
    # A GitHub link was given -- first see whether it's already sitting
    # in the local cache (master-folder-style layout: cache_dir/<repo
    # name>). If so, use that copy directly and skip the network clone
    # entirely. repo_url stays the original URL so downstream merging
    # (which keys on repo_url) behaves identically to a fresh clone.
    cached_path = _find_cached_repo(url, cache_dir)
    if cached_path:
        print(f"  Found cached copy of {url!r} at {cached_path!r} — skipping clone")
        return _load_single_local_repo(cached_path, repo_url=url)
    print(f"  No cached copy found in {cache_dir!r} — cloning {url!r} from GitHub")

    # ── Clone (remote URL) ───────────────────────────────────────────────
    clone_dir = _clone_repo(url, branch)

    try:
        all_files = _walk_repo(clone_dir)
        print(f"  Found {len(all_files)} supported source file(s) in repo")

        sources: list[dict] = []
        for fpath in all_files:
            lang = detect_language(fpath)
            rel  = os.path.relpath(fpath, clone_dir)   # path relative to repo root
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    code = fh.read()
            except OSError as exc:
                print(f"  ⚠  Could not read {rel!r}: {exc}")
                continue

            sources.append({
                "filepath": rel,
                "filename": os.path.basename(fpath),
                "repo_url": url,
                "repo_dir": clone_dir,
                "language": lang,
                "code":     code,
            })
            print(f"  Loaded  {rel!r}  [{lang}]")
    except Exception:
        # Something went wrong before we could hand the checkout off to
        # the caller (e.g. tree-sitter pass in json_loader.py) -- nobody
        # else knows about clone_dir, so it's safe (and necessary) to
        # clean it up right away here.
        shutil.rmtree(clone_dir, ignore_errors=True)
        print(f"  Deleted clone (after error) → {clone_dir}")
        raise

    # NOTE: clone_dir is deliberately NOT deleted here anymore. It's
    # needed on disk by json_loader.py's tree-sitter pass (extract_repo()
    # walks the real checkout), which runs *after* this function returns.
    # Register it for deferred cleanup -- call cleanup_temp_repos() once
    # that pass has actually used it.
    _TEMP_CLONE_DIRS.append(clone_dir)

    return sources