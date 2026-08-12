"""
finder.py
---------
Splits every per-repo JSON file under repo_outputs/ (one file per repo,
written by json_loader.analyse_all_files -- see that module's docstring)
into its individual record-type files, under output/<repo-name>/.

Usage:
    python finder.py                # split every file in repo_outputs/
    python finder.py metaflow.json  # split just one repo's output file
"""

import os
import sys
import json

REPO_OUTPUTS_DIR = "repo_outputs"
SPLIT_OUTPUT_DIR = "output"

KEYS = [
    "file_index",
    "function_call_flow",
    "data_flow",
    "variables_per_function",
    "class_diagram",
    "import_flow",
]


def split_one(json_path: str) -> None:
    repo_name = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.join(SPLIT_OUTPUT_DIR, repo_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{repo_name}  (from {json_path})")
    for key in KEYS:
        with open(os.path.join(out_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump(data.get(key, []), f, indent=4)
        print(f"  {key}: {len(data.get(key, []))} records")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args:
        targets = [
            a if os.path.dirname(a) else os.path.join(REPO_OUTPUTS_DIR, a)
            for a in args
        ]
    else:
        if not os.path.isdir(REPO_OUTPUTS_DIR):
            raise SystemExit(
                f"No {REPO_OUTPUTS_DIR!r} directory found -- run main.py first."
            )
        targets = sorted(
            os.path.join(REPO_OUTPUTS_DIR, f)
            for f in os.listdir(REPO_OUTPUTS_DIR)
            if f.endswith(".json")
        )

    if not targets:
        raise SystemExit(f"No repo JSON files found in {REPO_OUTPUTS_DIR!r}.")

    for path in targets:
        if not os.path.isfile(path):
            print(f"  ⚠  Skipping missing file: {path}")
            continue
        split_one(path)

    print("\nTask has been completed successfully.")