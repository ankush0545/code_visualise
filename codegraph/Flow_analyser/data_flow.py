"""
Flow_analyser.data_flow
-------------------------
DataFlowAnalyzer tracks, per function:
  - every simple-name variable it assigns (and its parameters)
  - every time one of those variables is passed as a positional
    argument into a call, and at what argument position

Interface expected by json_loader.py:
    analyzer = DataFlowAnalyzer()
    analyzer.visit(tree)
    analyzer.data_flows   # list[(function, variable, passed_to, arg_position)]
    analyzer.assignments  # dict[function -> list[str]]
                          #   params are prefixed "param:"
"""

import ast

from Flow_analyser._common import call_target_name, qualified_name


class DataFlowAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.data_flows: list[tuple[str, str, str, int]] = []
        self.assignments: dict[str, list[str]] = {}
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
        self.assignments.setdefault(name, [])

        for arg in node.args.args:
            if arg.arg != "self":
                self.assignments[name].append(f"param:{arg.arg}")

        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Assign(self, node):
        if self._func_stack:
            fn = self._func_stack[-1]
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assignments.setdefault(fn, []).append(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        if self._func_stack:
            fn = self._func_stack[-1]
            callee = call_target_name(node.func) or "<unknown>"
            for pos, arg in enumerate(node.args):
                if isinstance(arg, ast.Name):
                    self.data_flows.append((fn, arg.id, callee, pos))
        self.generic_visit(node)
