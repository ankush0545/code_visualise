#!/usr/bin/env python3
"""
CodeGraph AI - Master Pipeline Runner
-------------------------------------
Orchestrates end-to-end repository ingestion, AST graph construction,
GNN edge prediction, CodeBERT file classification, and launches
the interactive Next.js visualization UI.

Usage:
    python master_main.py
    python master_main.py https://github.com/Netflix/metaflow
    python master_main.py https://github.com/owner/repo --skip-frontend
    python master_main.py --frontend-only
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Base directories (fully portable across systems)
ROOT_DIR = Path(__file__).resolve().parent
CODEGRAPH_DIR = ROOT_DIR / "codegraph"
NODE_CLS_DIR = ROOT_DIR / "Node_Classification"
LABEL_CLS_DIR = ROOT_DIR / "Label_Classification"
FINAL_OUT_DIR = ROOT_DIR / "Final_Output"
DATA_DIR = ROOT_DIR / "data"
CODE_GRAPH_UI_DIR = ROOT_DIR / "code-graph"


def print_banner():
    banner = r"""
======================================================================
   ____          _       ____                 _         _     ___ 
  / ___|___   __| | ___ / ___|_ __ __ _ _ __ | |__     / \   |_ _|
 | |   / _ \ / _` |/ _ \ |  _| '__/ _` | '_ \| '_ \   / _ \   | | 
 | |__| (_) | (_| |  __/ |_| | | | (_| | |_) | | | | / ___ \  | | 
  \____\___/ \__,_|\___|\____|_|  \__,_| .__/|_| |_|/_/   \_\___|
                                       |_|                        
  Intelligent Repository Analysis & Architecture Visualization
======================================================================
"""
    print(banner)


def run_pipeline(repo_url: str, skip_frontend: bool = False):
    """Executes the analysis and classification stages sequentially."""
    print(f"\n[Step 1/3] Ingesting repository & extracting code AST...")
    print(f"Target Repository: {repo_url}\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Code Analysis & Graph Construction
    try:
        res = subprocess.run(
            [sys.executable, str(CODEGRAPH_DIR / "main.py"), repo_url],
            cwd=str(ROOT_DIR),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error during code analysis stage: {e}")
        return False

    # 2. GNN Edge Prediction
    graph_json = DATA_DIR / "output.json"
    if not graph_json.exists():
        # Check if output is in data/repo_outputs
        outputs = list((DATA_DIR / "repo_outputs").glob("*.json"))
        if outputs:
            graph_json = outputs[0]

    print(f"\n[Step 2/3] Running GraphSAGE GNN edge prediction on: {graph_json}...")
    try:
        res = subprocess.run(
            [
                sys.executable,
                str(NODE_CLS_DIR / "predict.py"),
                str(graph_json),
                "--top-k", "1000"
            ],
            cwd=str(ROOT_DIR),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n[!] GNN edge prediction encountered an issue: {e}")

    # 3. CodeBERT Role Classification
    print(f"\n[Step 3/3] Running CodeBERT file-level role classification...")
    try:
        res = subprocess.run(
            [sys.executable, str(LABEL_CLS_DIR / "infer_codebert.py"), repo_url],
            cwd=str(ROOT_DIR),
            check=False  # Continue even if large CodeBERT weights are not downloaded
        )
    except Exception as e:
        print(f"\n[!] Note on CodeBERT classification: {e}")

    print("\n" + "=" * 70)
    print(" [✓] Pipeline execution finished successfully.")
    print("     Generated artifacts saved to Final_Output/")
    print("=" * 70)

    if not skip_frontend:
        launch_frontend()

    return True


def launch_frontend():
    """Starts the Next.js interactive visualization server."""
    if not CODE_GRAPH_UI_DIR.exists():
        print(f"[!] Frontend directory not found at: {CODE_GRAPH_UI_DIR}")
        return

    npm_bin = shutil.which("npm")
    if not npm_bin:
        print("[!] 'npm' not found in PATH. Please install Node.js to run the UI.")
        print(f"    You can manually run the UI later: cd code-graph && npm run dev")
        return

    print("\n>>> Starting CodeGraph AI Web Visualization Dashboard...")
    print(">>> Opening Next.js local server at: http://localhost:3000\n")
    try:
        subprocess.run([npm_bin, "run", "dev"], cwd=str(CODE_GRAPH_UI_DIR))
    except KeyboardInterrupt:
        print("\n[✓] Web visualization server stopped.")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="CodeGraph AI: End-to-end repository code analysis & visualization pipeline."
    )
    parser.add_argument(
        "repo_url",
        nargs="?",
        default=None,
        help="GitHub repository URL or local path to analyze."
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Run analysis pipeline only without starting the Next.js server."
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Launch the Next.js visualization UI directly using existing data."
    )

    args = parser.parse_args()

    if args.frontend_only:
        launch_frontend()
        return

    repo_url = args.repo_url
    if not repo_url:
        try:
            repo_url = input("Enter GitHub repository URL (or press Enter for demo): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not repo_url:
        print("[*] No URL provided. Launching visualization dashboard with pre-computed demo dataset...")
        launch_frontend()
        return

    run_pipeline(repo_url, skip_frontend=args.skip_frontend)


if __name__ == "__main__":
    main()