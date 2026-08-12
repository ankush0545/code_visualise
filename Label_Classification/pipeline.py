"""
pipeline.py
-----------
End-to-end driver for the CodeBERT file-classifier pipeline, built on the
existing run.py batch-cloner over github_links.

Stages (in order):
  1. run       run.py           -- clone+classify every repo in github_links
                                    (batches of 5 via main.py), each writing
                                    Prediction_<repo>.json, using a
                                    PERSISTENT --cache-dir so the clones
                                    survive for stage 3 to reuse.
  2. merge     merge_predictions.py -- combine all Prediction_*.json files
                                    into one merged_predictions.json.
  3. dataset   build_dataset.py  -- turn merged_predictions.json + the
                                    cached repos into train/val/test.jsonl
                                    + labels.json.
  4. train     train_codebert    -- fine-tune CodeBERT on that dataset.
                                    Needs internet access to huggingface.co
                                    -- run this stage on a machine/CI that
                                    has it if this sandbox blocks it.
  5. infer     infer_codebert    -- run the fine-tuned model back over the
                                    cached repos to produce fresh, model-
                                    based predictions in the same schema.

Each stage runs as its own subprocess, with stdout/stderr both streamed
live and saved to pipeline_logs/<stage>.log. The pipeline stops at the
first failing stage and prints the exact command to resume from there.

Usage:
    python pipeline.py                          # run every stage
    python pipeline.py --skip run --skip merge  # resume from build_dataset
    python pipeline.py --only infer             # just re-run inference
"""
import argparse
import os
import subprocess
import sys

LOG_DIR = "pipeline_logs"
os.makedirs(LOG_DIR, exist_ok=True)

CACHE_DIR = os.path.expanduser("~/repo_cache")  # must match run.py's CACHE_DIR

STAGES = [
    ("run",     [sys.executable, "run.py"]),
    ("merge",   [sys.executable, "merge_predictions.py"]),
    ("dataset", [sys.executable, "build_dataset.py"]),
    ("train",   [sys.executable, "train_codebert.py"]),
    ("infer",   [sys.executable, "infer_codebert.py", "--master-folder", CACHE_DIR]),
]
STAGE_NAMES = [name for name, _ in STAGES]


def run_stage(name, command):
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
    print(" ".join(command))
    print(f"Logging to {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        process.wait()

    if process.returncode != 0:
        print(f"✗ Stage {name!r} failed (exit {process.returncode}) -- see {log_path}")
    else:
        print(f"✓ Stage {name!r} completed -- see {log_path}")
    return process.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[], choices=STAGE_NAMES,
                     help="skip a stage (repeatable) -- use to resume a partial run")
    ap.add_argument("--only", choices=STAGE_NAMES, help="run just one stage")
    args = ap.parse_args()

    if args.only:
        stages = [(n, c) for n, c in STAGES if n == args.only]
    else:
        stages = [(n, c) for n, c in STAGES if n not in args.skip]

    for i, (name, command) in enumerate(stages):
        rc = run_stage(name, command)
        if rc != 0:
            already_done = [n for n, _ in stages[:i]]
            skip_args = " ".join(f"--skip {n}" for n in already_done + [name])
            print(f"\nPipeline stopped at stage {name!r}. Resume with:\n"
                  f"  python pipeline.py {skip_args}")
            sys.exit(rc)

    print("\nPipeline finished: fine-tuned CodeBERT model ready, fresh predictions written.")


if __name__ == "__main__":
    main()