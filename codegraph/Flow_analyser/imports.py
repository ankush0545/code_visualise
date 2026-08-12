"""
Traces EXACT flow: which specific function uses which specific import,
not just "this file imports X somewhere".
"""
import ast
import os
import json
import networkx as nx

PROJECT_DIR = "import.py"


def get_import_aliases(tree):
    """
    Build a lookup: name-used-in-code -> real module.
    e.g. `import numpy as np`        -> {"np": "numpy"}
         `from pkg.helpers import do_thing` -> {"do_thing": "pkg.helpers"}
         `import os`                 -> {"os": "os"}
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                aliases[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local_name = alias.asname or alias.name
                aliases[local_name] = module  # imported straight into local namespace
    return aliases


def find_usages_in_function(func_node, aliases):
    """
    Walk everything INSIDE this one function and see which imported
    names actually get referenced.
    """
    used = set()
    for node in ast.walk(func_node):
        name_to_check = None
        if isinstance(node, ast.Name):
            name_to_check = node.id
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name_to_check = node.value.id  # e.g. requests.get -> checks "requests"

        if name_to_check and name_to_check in aliases:
            used.add(aliases[name_to_check])
    return used


def build_exact_flow_graph(project_dir):
    DG = nx.DiGraph()

    for root, _, files in os.walk(project_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, project_dir)

            with open(full_path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=full_path)
                except SyntaxError:
                    continue

            aliases = get_import_aliases(tree)

            # walk every function defined in this file
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_id = f"{rel_path}::{node.name}"   # unique id per function
                    DG.add_node(func_id, node_type="function", file=rel_path)

                    used_modules = find_usages_in_function(node, aliases)
                    for module in used_modules:
                        DG.add_node(module, node_type="module")
                        DG.add_edge(func_id, module)

    return DG


def export_for_visualization(DG, out_path="graph_data.json"):
    degree = dict(DG.degree())
    nodes = []
    for n, data in DG.nodes(data=True):
        color = "#f76f4f" if data.get("node_type") == "module" else "#4f8ef7"
        nodes.append({
            "id": n, "label": n.split("::")[-1], "value": degree.get(n, 1),
            "title": n, "color": color,
        })
    edges = [{"from": u, "to": v, "arrows": "to"} for u, v in DG.edges()]

    with open(out_path, "w") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
    print(f"{len(nodes)} nodes, {len(edges)} exact-usage edges -> {out_path}")


if __name__ == "__main__":
    graph = build_exact_flow_graph(PROJECT_DIR)
    export_for_visualization(graph)