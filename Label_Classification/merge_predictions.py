"""
merge_predictions.py
---------------------
Connects to EXISTING outputs:
  - A directory of per-repo Prediction_<repo>.json files, one per repo in
    run.py's github_links, written by main.py during run.py's batched run.
    Each file matches the existing schema (see Prediction_metaflow.json):
        [{"filepath", "repo_url", "predicted_label", "confidence",
          "is_confident", "all_labels"}, ...]

Produces NEW files:
  - merged_predictions.json -- every repo's predictions concatenated and
    de-duplicated by (repo_url, filepath), for build_dataset.py to consume.
  - merged_predictions_with_stats.json -- includes additional statistics
    and label distribution info

UPDATES:
  - Added compatibility with label remapping pipeline
  - Preserves original labels for traceability
  - Adds statistics helpful for dataset construction
  - Option to filter by confidence threshold

Usage:
    python merge_predictions.py
    python merge_predictions.py --dir /path/to/predictions --out /path/to/merged.json
    python merge_predictions.py --min-confidence 0.7 --out merged_filtered.json
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

DEFAULT_DIR = os.path.expanduser("~/work/predictions")
DEFAULT_OUT = os.path.expanduser("~/work/merged_predictions.json")
DEFAULT_STATS_OUT = os.path.expanduser("~/work/merged_predictions_stats.json")


def analyze_predictions(preds):
    """Analyze prediction distribution and quality"""
    total = len(preds)
    confident = sum(1 for p in preds if p.get("is_confident", False))
    
    label_counts = Counter(p["predicted_label"] for p in preds)
    confidence_by_label = defaultdict(list)
    for p in preds:
        confidence_by_label[p["predicted_label"]].append(p.get("confidence", 0))
    
    avg_confidence_by_label = {
        label: sum(vals) / len(vals) for label, vals in confidence_by_label.items()
    }
    
    # Track repos
    repos = {p.get("repo_url") for p in preds}
    
    return {
        "total_records": total,
        "confident_records": confident,
        "non_confident_records": total - confident,
        "unique_repos": len(repos),
        "label_distribution": dict(label_counts),
        "avg_confidence_by_label": avg_confidence_by_label,
        "labels": sorted(label_counts.keys()),
        "confidence_thresholds": {
            ">0.9": sum(1 for p in preds if p.get("confidence", 0) >= 0.9),
            ">0.8": sum(1 for p in preds if p.get("confidence", 0) >= 0.8),
            ">0.7": sum(1 for p in preds if p.get("confidence", 0) >= 0.7),
            ">0.6": sum(1 for p in preds if p.get("confidence", 0) >= 0.6),
            ">0.5": sum(1 for p in preds if p.get("confidence", 0) >= 0.5),
            ">0.4": sum(1 for p in preds if p.get("confidence", 0) >= 0.4),
            ">0.3": sum(1 for p in preds if p.get("confidence", 0) >= 0.3),
            ">0.2": sum(1 for p in preds if p.get("confidence", 0) >= 0.2),
            ">0.1": sum(1 for p in preds if p.get("confidence", 0) >= 0.1),
        }
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR,
                     help=f"directory containing per-repo Prediction_*.json files (default: {DEFAULT_DIR})")
    ap.add_argument("--pattern", default="Prediction_*.json",
                     help="glob pattern (relative to --dir) matching per-repo prediction files")
    ap.add_argument("--out", default=DEFAULT_OUT,
                     help=f"path for merged predictions JSON (default: {DEFAULT_OUT})")
    ap.add_argument("--stats-out", default=DEFAULT_STATS_OUT,
                     help=f"path for statistics JSON (default: {DEFAULT_STATS_OUT})")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                     help="minimum confidence threshold to include predictions (default: 0.0 = all)")
    ap.add_argument("--require-confident", action="store_true",
                     help="only include predictions marked as confident")
    args = ap.parse_args()

    # Find all prediction files
    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        raise SystemExit(
            f"No prediction files matched {args.pattern!r} in {args.dir!r}.\n"
            f"Check that main.py actually wrote its output there -- "
            f"pass --dir/--pattern if it uses a different location or naming."
        )

    print(f"Found {len(paths)} prediction file(s) in {args.dir!r}")
    
    merged = []
    seen = set()
    per_repo_counts = Counter()
    dupes = 0
    filtered_out = 0
    empty_files = 0
    parse_errors = 0

    # Load and merge all predictions
    for path in paths:
        try:
            with open(path) as f:
                preds = json.load(f)
        except json.JSONDecodeError:
            print(f"  WARNING: Could not parse {path} - skipping")
            parse_errors += 1
            continue
        
        if not preds:
            print(f"  WARNING: Empty file {path} - skipping")
            empty_files += 1
            continue
            
        for p in preds:
            # Apply filters
            if args.require_confident and not p.get("is_confident", False):
                filtered_out += 1
                continue
            if p.get("confidence", 0) < args.min_confidence:
                filtered_out += 1
                continue
                
            key = (p.get("repo_url"), p["filepath"])
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            merged.append(p)
            per_repo_counts[p.get("repo_url", "?")] += 1

    # Write merged predictions
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)

    # Generate and write statistics
    stats = analyze_predictions(merged)
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nMerged {len(paths)} file(s) -> {len(merged)} record(s)")
    print(f"  - {dupes} duplicate(s) skipped")
    print(f"  - {filtered_out} record(s) filtered out (confidence/confident flag)")
    if empty_files:
        print(f"  - {empty_files} empty file(s) skipped")
    if parse_errors:
        print(f"  - {parse_errors} file(s) had parse errors")
    
    print(f"\nRepos: {len(per_repo_counts)}")
    for repo, n in sorted(per_repo_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:5d}  {repo}")
    if len(per_repo_counts) > 10:
        print(f"  ... and {len(per_repo_counts) - 10} more repo(s)")

    print(f"\nLabels: {len(stats['labels'])} unique labels")
    print(f"Confident records: {stats['confident_records']}/{stats['total_records']}")
    
    print(f"\nLabel distribution (top 10):")
    label_dist = stats['label_distribution']
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1])[:10]:
        avg_conf = stats['avg_confidence_by_label'][label]
        print(f"  {label:22s} {count:6d}  (avg conf: {avg_conf:.3f})")
    if len(label_dist) > 10:
        print(f"  ... and {len(label_dist) - 10} more labels")
    
    print(f"\nConfidence thresholds (records with confidence >= X):")
    for threshold, count in stats['confidence_thresholds'].items():
        pct = count/len(merged)*100 if merged else 0
        print(f"  {threshold:10s} {count:6d} ({pct:.1f}%)")
    
    print(f"\nWrote merged predictions -> {args.out}")
    print(f"Wrote statistics -> {args.stats_out}")


if __name__ == "__main__":
    main()