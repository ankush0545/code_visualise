"""
Flow_analyser.function_flow
----------------------------
CallFlowAnalyzer walks a module and records, for every function or
method, every call made directly inside its body (not inside nested
functions defined within it — those get attributed to themselves).

Interface expected by json_loader.py:
    analyzer = CallFlowAnalyzer()
    analyzer.visit(tree)
    analyzer.calls   # list[(caller: str, callee: str)]
"""

import ast

from Flow_analyser._common import call_target_name, qualified_name


class CallFlowAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self._func_stack: list[str] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def _visit_func(self, node):
        name = qualified_name(self._class_stack, node.name)
        self._func_stack.append(name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Call(self, node):
        if self._func_stack:
            callee = call_target_name(node.func)
            if callee:
                self.calls.append((self._func_stack[-1], callee))
        self.generic_visit(node)
