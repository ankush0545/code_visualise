"""
Connects to EXISTING outputs:
  - /home/claude/work/merged_predictions.json  (from merge_predictions.py,
    itself merging the per-repo Prediction_<repo>.json files main.py wrote
    for every repo in run.py's github_links)
  - the persistent repo cache at CACHE_DIR (populated by main.py when run
    with the --cache-dir flag run.py now passes) -- each repo's source is
    read from here rather than being re-cloned from GitHub.

Produces NEW files:
  - train.jsonl / val.jsonl / test.jsonl  (schema: {"filepath", "text", "label"})
  - labels.json

NOW WITH LABEL REMAPPING: Converts from the old 9-label taxonomy to the new 16-label taxonomy
using the same logic as remap_labels.py
"""
import json
import os
import random
import re
from collections import defaultdict

from source_loader import _find_cached_repo, _repo_name_from_url, load_sources, cleanup_temp_repos

MERGED_PRED_FILE = os.path.expanduser("~/work/merged_predictions.json")
CACHE_DIR = os.path.expanduser("~/repo_cache")
OUT_DIR = os.path.expanduser("~/work/dataset")
MAX_CHARS = 2000
# FIXED: Lower threshold to get more examples for training
MIN_CONFIDENCE = 0.3  # Changed from 0.5 to 0.3
# FIXED: Allow non-confident predictions too for bootstrapping
REQUIRE_CONFIDENT = False  # Changed from True to False
SEED = 13

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

# ====== LABEL REMAPPING LOGIC (from remap_labels.py) ======
DIRECT_RENAME = {
    "test": "testing",
    "documentation": "documentation",
    "configuration": "configuration",
    "build_tooling": "build_tooling",
    "utility": "utility",
    "vendored_dependency": "vendored_dependency",
    "entry_point": "entry_point",
    "example": "example",
}

# Order matters: first matching rule wins
PATH_RULES = [
    # -----------------------------
    # Authentication / Authorization
    # -----------------------------
    (
        "auth",
        r"(^|/)(auth|authentication|authorization|authn|authz|oauth|oauth2|jwt|"
        r"session|sessions|login|signup|rbac|permissions?)(/|$)",
        None,
    ),

    # -----------------------------
    # Mobile
    # -----------------------------
    (
        "mobile",
        r"(^|/)(ios|android|mobile)(/|$)|"
        r"Podfile|\.xcodeproj|\.xcworkspace|"
        r"build\.gradle(\.kts)?|AndroidManifest\.xml|"
        r"\.(swift|kt|kts|m|mm)$",
        None,
    ),

    # -----------------------------
    # Infrastructure / DevOps
    # -----------------------------
    (
        "infra_devops",
        r"(^|/)(\.github/workflows|\.gitlab-ci|infra|terraform|k8s|"
        r"kubernetes|helm|ansible|deploy|ops|iac)(/|$)|"
        r"(^|/)(Dockerfile|docker-compose(\.ya?ml)?)$",
        None,
    ),

    # -----------------------------
    # Database / Persistence
    # -----------------------------
    (
        "database",
        r"(^|/)(migrations?|schema|schemas|seeds?|orm|"
        r"repositories?|dao|persistence|storage)(/|$)|"
        r"\.(sql)$",
        None,
    ),

    # -----------------------------
    # API layer
    # -----------------------------
    (
        "api",
        r"(^|/)(api|routes?|controllers?|endpoints?|"
        r"graphql|rpc|rest|gateway|resolvers?)(/|$)",
        None,
    ),

    # -----------------------------
    # Machine Learning / Data
    # -----------------------------
    (
        "ml_data",
        r"(^|/)(datasets?|notebooks?|training|preprocessing|"
        r"features?|pipelines?|experiments?|embeddings?|"
        r"rag|retrieval|evaluation|fine_tuning|"
        r"feature_store|inference)(/|$)|"
        r"\.(ipynb)$",
        None,
    ),

    # -----------------------------
    # Frontend
    # -----------------------------
    (
        "frontend",
        r"(^|/)(components?|pages?|views?|screens?|widgets?|"
        r"layouts?|hooks?|store|stores?|assets|public|"
        r"styles?|themes?|frontend|web|client|app)(/|$)|"
        r"\.(jsx|tsx|vue|svelte|css|scss|less|html)$",
        None,
    ),

    # -----------------------------
    # Backend business logic
    # -----------------------------
    (
        "backend",
        r"(^|/)(backend|server|services?|handlers?|"
        r"middleware|workers?|jobs?|consumers?|"
        r"producers?|processors?)(/|$)",
        None,
    ),

    # -----------------------------
    # Utility
    # -----------------------------
    (
        "utility",
        r"(^|/)(utils?|helpers?|common|shared|lib|libs?|"
        r"constants?|types?)(/|$)",
        None,
    ),
]

CONTENT_RULES = [
    (
        "auth",
        r"\b(jwt|oauth2?|passport|bcrypt|authenticate|"
        r"login_required|permission_classes)\b",
    ),

    (
        "mobile",
        r"\b(UIViewController|SwiftUI|androidx\.|"
        r"Activity\s*:|Fragment|ReactNative|expo-)\b",
    ),

    (
        "frontend",
        r"\b(useState|useEffect|useMemo|useCallback|"
        r"React\.Component|document\.querySelector|"
        r"createElement|NextPage|defineComponent)\b",
    ),

    (
        "database",
        r"\b(sqlalchemy|sequelize|typeorm|prisma|"
        r"mongoose\.Schema|CREATE TABLE|ALTER TABLE|"
        r"SELECT\s+.+\s+FROM)\b",
    ),

    (
        "api",
        r"\b("
        r"@app\.route|APIRouter|fastapi\.|"
        r"express\.Router|graphql|"
        r"@RestController|@GetMapping|@PostMapping|"
        r"gin\.Default|fiber\.New|"
        r"\[ApiController\]"
        r")\b",
    ),

    (
        "infra_devops",
        r"\b("
        r"kubectl|docker build|"
        r"apiVersion:|kind:|"
        r"resource\s+\"aws_|"
        r"terraform|ansible"
        r")\b",
    ),

    (
        "ml_data",
        r"\b("
        r"torch\.nn|tensorflow|keras|"
        r"sklearn|xgboost|lightgbm|"
        r"pandas\.DataFrame|"
        r"model\.fit\(|DataLoader|"
        r"transformers\."
        r")\b",
    ),

    (
        "backend",
        r"\b("
        r"class .*Service|"
        r"@Injectable|"
        r"async def|"
        r"class .*Handler|"
        r"class .*Worker"
        r")\b",
    ),
]
def remap_label(old_label, filepath, content):
    """Apply the same remapping logic as remap_labels.py"""
    if old_label in DIRECT_RENAME:
        return DIRECT_RENAME[old_label], "direct_rename"
    elif old_label == "core_logic":
        # Extract just the filepath part for matching
        path_part = filepath.split("/", 1)[-1] if "/" in filepath else filepath
        
        # Try path-based rules first
        for new_label, path_pat, _ in PATH_RULES:
            if re.search(path_pat, path_part, re.IGNORECASE):
                return new_label, "path"
        
        # Then content-based rules
        for new_label, content_pat in CONTENT_RULES:
            if re.search(content_pat, content, re.IGNORECASE):
                return new_label, "content"
        
        # Default fallback
        return "backend", "fallback_default"
    else:
        # Unknown label - keep as-is
        return old_label, "unknown_label_passthrough"

# ====== MAIN DATASET BUILDING LOGIC ======

def _repo_dir_for(url):
    """Locate this repo's source on disk."""
    cached = _find_cached_repo(url, CACHE_DIR)
    if cached:
        return cached, False
    print(f"  {url!r} not found in cache ({CACHE_DIR!r}) -- cloning fresh")
    sources = load_sources([url])
    if not sources:
        return None, False
    return sources[0]["repo_dir"], True

def load_and_remap_examples():
    """Load predictions and remap labels according to new taxonomy"""
    # Check if merged file exists
    if not os.path.exists(MERGED_PRED_FILE):
        print(f"ERROR: {MERGED_PRED_FILE} not found!")
        print("Run merge_predictions.py first.")
        return []
    
    with open(MERGED_PRED_FILE) as f:
        preds = json.load(f)
    
    print(f"Loaded {len(preds)} total predictions from merged file")

    by_repo = defaultdict(list)
    filtered_out = 0
    for p in preds:
        # FIXED: More flexible filtering
        should_include = False
        if REQUIRE_CONFIDENT:
            should_include = p.get("is_confident", False) and p.get("confidence", 0) >= MIN_CONFIDENCE
        else:
            # Include if confidence >= threshold, regardless of is_confident flag
            should_include = p.get("confidence", 0) >= MIN_CONFIDENCE
        
        if should_include:
            by_repo[p["repo_url"]].append(p)
        else:
            filtered_out += 1
    
    print(f"Filtered out {filtered_out} predictions (confidence < {MIN_CONFIDENCE})")
    print(f"Keeping {sum(len(v) for v in by_repo.values())} predictions from {len(by_repo)} repos")

    examples = []
    skipped_missing = 0
    used_temp_clone = False
    remap_stats = defaultdict(int)
    repo_stats = {}

    for repo_url, repo_preds in by_repo.items():
        print(f"\nProcessing repo: {repo_url} ({len(repo_preds)} predictions)")
        repo_dir, is_temp = _repo_dir_for(repo_url)
        used_temp_clone = used_temp_clone or is_temp
        if repo_dir is None:
            print(f"  WARNING: Could not find repo {repo_url} in cache")
            skipped_missing += len(repo_preds)
            continue
        
        repo_name = _repo_name_from_url(repo_url)
        found_files = 0
        missing_files = 0
        
        for p in repo_preds:
            full_path = os.path.join(repo_dir, p["filepath"])
            try:
                with open(full_path, "r", errors="ignore") as f:
                    content = f.read(MAX_CHARS)
                found_files += 1
            except OSError:
                missing_files += 1
                continue

            qualified_path = f"{repo_name}/{p['filepath']}"
            
            # Apply label remapping
            old_label = p["predicted_label"]
            new_label, method = remap_label(old_label, qualified_path, content)
            remap_stats[f"{old_label}->{new_label}"] += 1
            
            text = f"{qualified_path}\n{content}"
            examples.append({
                "filepath": qualified_path,
                "text": text,
                "label": new_label,
                "original_label": old_label,  # Keep for traceability
                "remap_method": method
            })
        
        repo_stats[repo_name] = {"found": found_files, "missing": missing_files}
        print(f"  Found {found_files} files, missing {missing_files}")

    if used_temp_clone:
        cleanup_temp_repos()

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total examples loaded: {len(examples)}")
    print(f"  Skipped (missing/unreadable): {skipped_missing}")
    print(f"  Repos processed: {len(repo_stats)}")
    
    if len(examples) == 0:
        print("\n⚠️  WARNING: No examples loaded! Try lowering MIN_CONFIDENCE further.")
        print(f"Current MIN_CONFIDENCE = {MIN_CONFIDENCE}")
        print(f"REQUIRE_CONFIDENT = {REQUIRE_CONFIDENT}")
    
    print("\nLabel remapping statistics (top 20):")
    for mapping, count in sorted(remap_stats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {mapping:35s} {count}")
    
    return examples

def group_key(filepath):
    """Group by repo + first module dir to prevent data leakage"""
    parts = filepath.split("/")
    return "/".join(parts[:3]) if len(parts) > 2 else filepath

def split(examples, train_frac=0.8, val_frac=0.1):
    """Split by groups to avoid leakage"""
    if not examples:
        return [], [], []
    
    groups = defaultdict(list)
    for ex in examples:
        groups[group_key(ex["filepath"])].append(ex)

    keys = list(groups.keys())
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])

    train = [ex for k in train_keys for ex in groups[k]]
    val = [ex for k in val_keys for ex in groups[k]]
    test = [ex for k in test_keys for ex in groups[k]]
    return train, val, test

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main():
    # Load and remap examples
    examples = load_and_remap_examples()
    
    if not examples:
        print("\n❌ ERROR: No examples loaded. Dataset generation failed.")
        print("Possible solutions:")
        print("1. Run merge_predictions.py first")
        print("2. Lower MIN_CONFIDENCE in build_dataset.py")
        print("3. Set REQUIRE_CONFIDENT = False")
        print("4. Check that predictions exist in ~/work/predictions/")
        return
    
    # Split into train/val/test
    train, val, test = split(examples)
    
    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Write output files
    os.makedirs(OUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUT_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val.jsonl"), val)
    write_jsonl(os.path.join(OUT_DIR, "test.jsonl"), test)

    # Get all labels in the new taxonomy
    labels = sorted({ex["label"] for ex in examples})
    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ DATASET GENERATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Output directory: {OUT_DIR}")
    print(f"  - train.jsonl: {len(train)} examples")
    print(f"  - val.jsonl:   {len(val)} examples")
    print(f"  - test.jsonl:  {len(test)} examples")
    print(f"  - labels.json: {len(labels)} labels")
    print(f"\nLabels: {labels}")
    
    # Print label distribution
    label_counts = defaultdict(int)
    for ex in examples:
        label_counts[ex["label"]] += 1
    print("\nLabel distribution (all splits combined):")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(examples) * 100
        print(f"  {label:22s} {count:6d} ({pct:.1f}%)")

if __name__ == "__main__":
    main()