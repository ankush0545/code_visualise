"""
Flow_analyser.class_flow
--------------------------
ClassDiagramAnalyzer records, per class:
  - base class names
  - attributes: class-level assignments/annotations + any self.x = ...
    assigned inside its methods
  - methods: signature strings like "method_name(self, a, b)"

Interface expected by json_loader.py:
    analyzer = ClassDiagramAnalyzer()
    analyzer.visit(tree)
    analyzer.classes  # dict[cls -> {"bases": [...], "attributes": [...], "methods": [...]}]
"""

import ast


def _base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class ClassDiagramAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.classes: dict[str, dict] = {}

    def visit_ClassDef(self, node):
        bases = [b for b in (_base_name(base) for base in node.bases) if b]
        attributes: set[str] = set()
        methods: list[str] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [a.arg for a in item.args.args]
                methods.append(f"{item.name}({', '.join(params)})")

                for sub in ast.walk(item):
                    if isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if (isinstance(t, ast.Attribute)
                                    and isinstance(t.value, ast.Name)
                                    and t.value.id == "self"):
                                attributes.add(t.attr)
                    elif isinstance(sub, ast.AnnAssign):
                        t = sub.target
                        if (isinstance(t, ast.Attribute)
                                and isinstance(t.value, ast.Name)
                                and t.value.id == "self"):
                            attributes.add(t.attr)

            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name):
                        attributes.add(t.id)

            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.add(item.target.id)

        self.classes[node.name] = {
            "bases": bases,
            "attributes": sorted(attributes),
            "methods": methods,
        }

        self.generic_visit(node)  # descend, in case of nested classes
