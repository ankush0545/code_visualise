"""
Tree-sitter based code analyzers for JavaScript, TypeScript, Go, Java,
C, and C++ -- output schema matches your existing Python/AST pipeline
(Flow_analyser.*) exactly, so it plugs directly into graph.py's
build_unified_graph() with no changes needed there.

    from treesitter_analyzers.extract_repo import extract_repo
    raw = extract_repo("./repo_cache/some-js-repo", repo_url="https://github.com/...")
"""
