"""
Connects to the model output produced by train_codebert.py.

train_codebert.py now trains in chunks across multiple Kaggle sessions, so
the "latest" model checkpoint may live in different places from run to run
(the current session's /kaggle/working, or an attached previous-run output
dataset under /kaggle/input). This script mirrors train_codebert.py's own
checkpoint-discovery logic: it looks for training_config.json (written at
the end of a training run) in every known candidate location and uses
whichever copy is newest, instead of assuming one fixed path.

Loads source files the SAME way classify.py does -- via source_loader.py's
load_sources() -- so inference accepts any repo URL, local checkout, cached
repo, or --master-folder of repos, and sees the identical file population
(same extensions, same skipped dirs) the model was trained on.

Produces output in the IDENTICAL schema to Prediction_metaflow.json / Prediction.json:
  filepath, repo_url, predicted_label, confidence, is_confident, all_labels

Usage
-----
    python Label_Classification/infer_codebert.py https://github.com/Netflix/metaflow
    python infer_codebert.py https://github.com/Netflix/metaflow --branch develop
    python infer_codebert.py /path/to/local/repo
    python infer_codebert.py --master-folder /path/to/master_folder
    python infer_codebert.py https://github.com/owner/repo --out /custom/path.json

Output defaults to:
  /home/claude/work/Prediction_<repo-name>_codebert.json
(or Prediction_multi_codebert.json for --master-folder, since that spans
several repos in one run)
"""
import glob
import json
import os
import sys
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding

from source_loader import load_sources, cleanup_temp_repos, _repo_name_from_url

OUT_DIR = "Final_Output"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_LEN = 512
MAX_CHARS = 2000
CONFIDENCE_THRESHOLD = 0.5   # same threshold used in classify.py's is_confident
BATCH_SIZE = 32

# ============================================================
# FIND TRAINED MODEL AUTOMATICALLY
# ============================================================
# Mirrors train_codebert.py's _find_previous_checkpoint(): a checkpoint is
# identified by a training_config.json sitting next to the model weights.
# codebert-file-classifier/ lives next to this script, so resolve it
# relative to the script's own location (not the current working directory,
# which changes depending on where infer_codebert.py is invoked from).
# Also check /kaggle/working and any attached /kaggle/input dataset in case
# this ever runs inside a Kaggle session directly.
_MODEL_DIR_SEARCH_PATHS = [
    os.path.join(SCRIPT_DIR, "codebert-file-classifier"),
    os.path.join(OUT_DIR, "codebert-file-classifier"),
    "/kaggle/working/codebert-file-classifier",
]


def _find_model_dir():
    candidates = []
    for path in _MODEL_DIR_SEARCH_PATHS:
        cfg = os.path.join(path, "training_config.json")
        if os.path.exists(cfg):
            candidates.append(cfg)
    candidates.extend(glob.glob("/kaggle/input/**/training_config.json", recursive=True))

    if not candidates:
        raise FileNotFoundError(
            "Could not find a trained model (no training_config.json found). "
            "Looked in: " + ", ".join(_MODEL_DIR_SEARCH_PATHS + ["/kaggle/input/**"])
        )

    candidates.sort(key=os.path.getmtime)
    model_dir = os.path.dirname(candidates[-1])
    print(f"Using model checkpoint: {model_dir}")
    return model_dir


MODEL_DIR = _find_model_dir()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)

# ============================================================
# LAYER-NORM KEY REMAP FIX (mirrors train_codebert.py)
# ============================================================
# Defensive: if this checkpoint was ever produced by a version of
# train_codebert.py from before the gamma/beta remap fix existed, its
# LayerNorm.weight/bias may actually have been randomly (re)initialized
# instead of trained, with the old TF-style keys never getting loaded.
# Detect and repair that the same way training does, rather than silently
# serving predictions from a partially-untrained model.

def _rename_gamma_beta(state_dict):
    renamed = {}
    for key, value in state_dict.items():
        new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight") \
                     .replace("LayerNorm.beta", "LayerNorm.bias")
        renamed[new_key] = value
    return renamed


def load_model_with_fixed_layernorm(model_path):
    model = RobertaForSequenceClassification.from_pretrained(model_path)

    safetensors_path = os.path.join(model_path, "model.safetensors")
    bin_path = os.path.join(model_path, "pytorch_model.bin")

    raw_state_dict = None
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file as load_safetensors
        raw_state_dict = load_safetensors(safetensors_path)
    elif os.path.exists(bin_path):
        raw_state_dict = torch.load(bin_path, map_location="cpu")

    if raw_state_dict is not None and any(
        "gamma" in k or "beta" in k for k in raw_state_dict.keys()
    ):
        print("⚠️ Old-style LayerNorm keys (gamma/beta) detected in checkpoint — remapping...")
        fixed_state_dict = _rename_gamma_beta(raw_state_dict)
        missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
        print(f"   After remap — missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            print(f"   Missing (first 5): {missing[:5]}")
        if unexpected:
            print(f"   Unexpected (first 5): {unexpected[:5]}")

    return model


model = load_model_with_fixed_layernorm(MODEL_DIR)
model.to(device).eval()
id2label = model.config.id2label

# Dynamic padding collator: pads each batch only to the longest sequence in
# that batch (rounded up to a multiple of 8), instead of always padding
# every example to MAX_LEN. Mirrors the DataCollatorWithPadding used for
# training, and is faster here too since most files are far shorter than
# MAX_LEN.
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    pad_to_multiple_of=8,
)


def predict_batch(items):
    """items: list of (filepath, content, repo_url) tuples. Tokenizes and
    runs the whole batch through the model in one forward pass, with
    per-batch dynamic padding instead of one file at a time."""
    encoded = [
        tokenizer(f"{filepath}\n{content}", truncation=True, max_length=MAX_LEN)
        for filepath, content, _ in items
    ]
    batch = data_collator(encoded)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for (filepath, _content, repo_url), row in zip(items, probs):
        ranked = sorted(
            [(id2label[i], float(p)) for i, p in enumerate(row)],
            key=lambda x: x[1], reverse=True,
        )
        top_label, top_conf = ranked[0]
        is_confident = top_conf >= CONFIDENCE_THRESHOLD
        all_labels = [{"label": l, "confidence": round(c, 4)} for l, c in ranked if c >= CONFIDENCE_THRESHOLD]

        results.append({
            "filepath": filepath,
            "repo_url": repo_url,
            "predicted_label": top_label,
            "confidence": round(top_conf, 4),
            "is_confident": is_confident,
            "all_labels": all_labels if is_confident else [],
        })
    return results


def _default_out_file(args):
    if "--master-folder" in args or "-m" in args:
        return os.path.join(OUT_DIR, "Prediction_multi_codebert.json")
    # First positional (non-flag) arg is the repo URL/local path
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("--branch", "-b", "--cache-dir", "-c", "--out"):
            skip_next = True
            continue
        if a.startswith("-"):
            continue
        name = _repo_name_from_url(a) if a.startswith(("http://", "https://")) else os.path.basename(os.path.normpath(a))
        return os.path.join(OUT_DIR, f"Prediction_codebert.json")
    return os.path.join(OUT_DIR, "Prediction_codebert.json")


def main():
    args = sys.argv[1:]
    if not args:
        raise ValueError(
            "No repository given.\n"
            "Usage:  python infer_codebert.py https://github.com/Netflix/metaflow.git"
            "        python infer_codebert.py /path/to/local/repo\n"
            "        python infer_codebert.py --master-folder /path/to/master_folder"
        )

    # --out is our own flag, not source_loader's -- strip it before handing
    # args off to load_sources(), which doesn't know about it.
    out_file = None
    if "--out" in args:
        idx = args.index("--out")
        out_file = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if out_file is None:
        out_file = _default_out_file(args)

    sources = load_sources(args)
    print(f"Loaded {len(sources)} source file(s) across {len({s['repo_url'] for s in sources})} repo(s)")

    try:
        results = []
        items = [(s["filepath"], s["code"][:MAX_CHARS], s["repo_url"]) for s in sources]
        for i in range(0, len(items), BATCH_SIZE):
            batch_items = items[i:i + BATCH_SIZE]
            results.extend(predict_batch(batch_items))
    finally:
        # Mirrors classify.py's contract with source_loader: any repo that
        # had to be freshly `git clone`'d is left on disk until we're done
        # reading from it, then cleaned up here. Cached/local/master-folder
        # repos are untouched by this call.
        cleanup_temp_repos()

    results.sort(key=lambda r: r["confidence"], reverse=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} entries to {out_file}")


if __name__ == "__main__":
    main()