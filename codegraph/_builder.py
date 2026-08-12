"""
pyvis_builder.py
----------------
Provides build_pyvis(): renders a networkx DiGraph as an interactive pyvis
HTML file, using the colors/options already defined in setting.py.

This function was called from main.py in four places but never defined
anywhere in the codebase — that was causing a NameError as soon as main.py
ran. This file fixes that.
"""

from pyvis.network import Network
import  setting


def build_pyvis(
    G,
    title: str,
    filename: str,
    gravity: int = -3000,
    spring_length: int = 200,
    use_display_label: bool = True,
) -> None:
    """
    Render graph G to an interactive HTML file using pyvis.

    - Node color comes from setting.PYVIS_COLORS, keyed by node_type
      (falls back to setting.START_COLOR_PYVIS / END_COLOR_PYVIS for
      start/end nodes, matching graph_builder._annotate_graph's role field).
    - Node label uses display_label (set by graph_builder._annotate_graph)
      when use_display_label=True, otherwise the raw node name.
    - Physics/interaction options come from setting.PYVIS_OPTIONS.
    - The zoom/fit toolbar from setting.ZOOM_JS is injected into the
      generated HTML before saving.
    """
    net = Network(height="900px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=gravity, central_gravity=0.2, spring_length=spring_length)

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "function")
        role = attrs.get("role")

        if role == "start":
            color = setting.START_COLOR_PYVIS
        elif role == "end":
            color = setting.END_COLOR_PYVIS
        else:
            color = setting.PYVIS_COLORS.get(node_type, "#CCCCCC")

        label = attrs.get("display_label", str(node)) if use_display_label else str(node)
        net.add_node(node, label=label, color=color, title=str(node))

    for source, target, attrs in G.edges(data=True):
        edge_label = attrs.get("label", "")
        net.add_edge(source, target, label=edge_label)

    net.set_options(setting.PYVIS_OPTIONS)
    net.write_html(filename, notebook=False)

    # Inject the zoom/fit-to-screen toolbar from setting.py
    with open(filename, "r", encoding="utf-8") as f:
        html = f.read()
    if "</body>" in html:
        html = html.replace("</body>", setting.ZOOM_JS + "\n</body>")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

    print(f"  ✓  {title} → {filename}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")