import networkx as nx
import  setting


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assign_sequence_numbers(G: nx.DiGraph) -> dict[str, int]:
    """
    Return {node: seq_number} where seq_number reflects execution order.
    Uses Kahn's topological sort; cycles are broken by falling back to
    in-degree order so every node always gets a number.
    """
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        # Graph has cycles — sort by in-degree as a best-effort ordering
        order = sorted(G.nodes(), key=lambda n: G.in_degree(n))
    return {node: i + 1 for i, node in enumerate(order)}



def _label(name: str, seq: int, role: str) -> str:
    """Build the display label shown on each node."""
    prefix = "▶ " if role == "start" else ("■ " if role == "end" else "")
    return f"{prefix}[{seq}] {name}"


def _annotate_graph(G: nx.DiGraph) -> None:
    """
    In-place: assign seq, role (start/end/mid), and display_label to every node.
    Start = no incoming edges.  End = no outgoing edges.
    """
    seq_map = _assign_sequence_numbers(G)

    for node in G.nodes():
        seq  = seq_map[node]
        is_start = G.in_degree(node) == 0
        is_end   = G.out_degree(node) == 0

        if is_start and is_end:        # isolated node — treat as start
            role = "start"
        elif is_start:
            role = "start"
        elif is_end:
            role = "end"
        else:
            role = "mid"

        G.nodes[node]["seq"]           = seq
        G.nodes[node]["role"]          = role
        G.nodes[node]["display_label"] = _label(node, seq, role)


# ── Color helpers for matplotlib ──────────────────────────────────────────────

def _node_colors(G: nx.DiGraph, base_color_map: dict[str, str],
                 default: str = "orange") -> list[str]:
    """
    Return a color list where start nodes are always 'limegreen',
    end nodes are always 'tomato', and others follow base_color_map.
    """
    colors = []
    for n in G.nodes():
        role = G.nodes[n].get("role", "mid")
        if role == "start":
            colors.append("limegreen")
        elif role == "end":
            colors.append("tomato")
        else:
            ntype = G.nodes[n].get("node_type", default)
            colors.append(base_color_map.get(ntype, default))
    return colors


# ============================================================
# 1. CALL GRAPH
# ============================================================
print("\n── Building Call Graph ──")


def build_call_graph(data):
    CG = nx.DiGraph()
    repo_functions = {item["function"] for item in data["function_call_flow"]}

    for item in data["function_call_flow"]:
        caller = item["function"]
        if caller not in CG.nodes:
            CG.add_node(caller, node_type="function")

        for callee in item["calls"]:
            if callee in repo_functions:
                ntype = "function"
            elif callee.startswith("self."):
                ntype = "method"
            elif callee in setting.BUILTINS:
                ntype = "builtin"
            else:
                ntype = "library"

            if callee not in CG.nodes:
                CG.add_node(callee, node_type=ntype)
            CG.add_edge(caller, callee)

    _annotate_graph(CG)
    print("  call_graph.html          – interactive call graph")
    print("  call_graph_static.png    – static call graph image")
    return CG


# ============================================================
# 2. DEPENDENCY GRAPH
# ============================================================
print("\n── Building Dependency Graph ──")


def build_dependency_graph(data):
    DG = nx.DiGraph()
    repo_functions = {item["function"] for item in data["function_call_flow"]}

    for item in data["function_call_flow"]:
        func = item["function"]
        if func not in DG.nodes:
            DG.add_node(func, node_type="function")

        seen_modules = set()
        for callee in item["calls"]:
            if callee in repo_functions:
                continue
            if callee.startswith("self."):
                continue
            if callee in setting.BUILTINS:
                continue

            parts  = callee.split(".")
            module = parts[0] if len(parts) > 1 else callee

            if module in setting.BUILTINS or module == "":
                continue

            if module not in seen_modules:
                seen_modules.add(module)
                if module not in DG.nodes:
                    DG.add_node(module, node_type="module")
                DG.add_edge(func, module)

    _annotate_graph(DG)
    print("  dependency_graph.html    – interactive dependency graph")
    print("  dependency_graph_static.png – static dependency graph image")
    return DG


# ============================================================
# 3. DATA FLOW GRAPH
# ============================================================
print("\n── Building Data Flow Graph ──")


def build_data_flow_graph(data):
    FG = nx.DiGraph()

    for item in data["data_flow"]:
        func      = item["function"]
        variable  = item["variable"]
        passed_to = item["passed_to"]
        arg_pos   = item["arg_position"]

        if func not in FG.nodes:
            FG.add_node(func, node_type="function")

        var_node = f"{func}::{variable}"
        if var_node not in FG.nodes:
            FG.add_node(var_node, node_type="variable", label=variable)

        if passed_to not in FG.nodes:
            FG.add_node(passed_to, node_type="callee")

        if not FG.has_edge(func, var_node):
            FG.add_edge(func, var_node, edge_type="owns")

        if not FG.has_edge(var_node, passed_to):
            FG.add_edge(var_node, passed_to,
                        edge_type="passed_as", label=f"arg {arg_pos}")

    _annotate_graph(FG)
    print("  data_flow_graph.html        – interactive data flow graph")
    print("  data_flow_graph_static.png  – static data flow graph image")
    return FG


# ============================================================
# 4. CLASS DIAGRAM GRAPH
# ============================================================
print("\n── Building Class Diagram Graph ──")


def build_class_diagram_graph(data):
    CDG = nx.DiGraph()

    for item in data["class_diagram"]:
        cls        = item["class"]
        bases      = item["bases"]
        attributes = item["attributes"]
        methods    = item["methods"]

        if cls not in CDG.nodes:
            CDG.add_node(cls, node_type="class")

        for base in bases:
            if base not in CDG.nodes:
                CDG.add_node(base, node_type="base_class")
            CDG.add_edge(cls, base, edge_type="inherits")

        for attr in attributes:
            attr_node = f"{cls}.{attr}"
            if attr_node not in CDG.nodes:
                CDG.add_node(attr_node, node_type="attribute", label=attr)
            CDG.add_edge(cls, attr_node, edge_type="has_attribute")

        for method_sig in methods:
            method_name = method_sig.split("(")[0]
            method_node = f"{cls}.{method_name}"
            if method_node not in CDG.nodes:
                CDG.add_node(method_node, node_type="method", label=method_sig)
            CDG.add_edge(cls, method_node, edge_type="has_method")

    _annotate_graph(CDG)
    print("  class_diagram.html        – interactive class diagram")
    print("  class_diagram_static.png  – static class diagram image")
    return CDG


# ============================================================
# COLOR LIST HELPERS  (used in main.py)
# ============================================================

def cg_node_colors(CG):
    return _node_colors(CG, setting.MPL_COLORS, "orange")

def dg_node_colors(DG):
    return _node_colors(DG, setting.MPL_COLORS, "orange")

def fg_node_colors(FG):
    return _node_colors(FG, setting.MPL_COLORS, "orange")

def cdg_node_colors(CDG):
    return _node_colors(CDG, setting.MPL_COLORS, "mediumpurple")

