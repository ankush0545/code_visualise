"""
Flow_analyser._common
----------------------
Small shared helpers used by function_flow.py, data_flow.py and
class_flow.py so the three analysers stay consistent with each other
(and with imports.py, which needs the *same* qualified function names
to join its import-usage data onto function_call_flow records).
"""

import ast


def qualified_name(class_stack: list[str], func_name: str) -> str:
    """'ClassName.method' if inside a class, else just 'func_name'."""
    return f"{class_stack[-1]}.{func_name}" if class_stack else func_name


def call_target_name(func_node: ast.AST) -> str | None:
    """
    Resolve the dotted name being called from a Call node's `.func`.
      foo()            -> "foo"
      self.bar()        -> "self.bar"
      requests.get()     -> "requests.get"
      a.b.c()            -> "a.b.c"
      foo()()            -> None (can't name a call result)
    """
    if isinstance(func_node, ast.Name):
        return func_node.id

    if isinstance(func_node, ast.Attribute):
        parts = []
        cur = func_node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None  # base of the chain wasn't a simple name (e.g. foo().bar)

    return None


def iter_qualified_functions(tree: ast.AST):
    """
    Yield (qualified_name, func_node) for every function/method defined
    anywhere in the tree, including nested functions and methods inside
    (possibly nested) classes. This is the canonical walk every analyser
    uses so that "MyClass.my_method" means the same thing everywhere.
    """
    def _walk(node, class_stack):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from _walk(child, class_stack + [child.name])
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield qualified_name(class_stack, child.name), child
                yield from _walk(child, class_stack)  # nested defs
            else:
                yield from _walk(child, class_stack)

    yield from _walk(tree, [])
