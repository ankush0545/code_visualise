"""
treesitter_analyzers.data_flow_ts
------------------------------------
Multi-language equivalent of Flow_analyser.data_flow.DataFlowAnalyzer.
Same output shape:

    analyzer = DataFlowAnalyzerTS(lang)
    analyzer.analyze(tree, source_bytes)
    analyzer.data_flows   # list[(function, variable, passed_to, arg_position)]
    analyzer.assignments  # dict[function -> list[str]]  (params prefixed "param:")

Only tracks SIMPLE identifier arguments (a bare variable name passed
positionally into a call) -- same restriction as the Python version.
Keyword/named arguments, expressions, and literals are intentionally
not traced here; that's a deliberate scope match with the original,
not an oversight.
"""

from __future__ import annotations

from .common import qualified_name, node_text
from .lang_config import LANG_CONFIG
from . import node_ops

_SIMPLE_ID_TYPES = {
    "javascript": "identifier", "typescript": "identifier", "tsx": "identifier",
    "go": "identifier", "java": "identifier", "c": "identifier", "cpp": "identifier",
}

# (assignment node type, field holding the assignment target)
_ASSIGN_TARGET_FIELDS = {
    "javascript": [("variable_declarator", "name"), ("assignment_expression", "left")],
    "typescript": [("variable_declarator", "name"), ("assignment_expression", "left")],
    "tsx": [("variable_declarator", "name"), ("assignment_expression", "left")],
    "go": [("short_var_declaration", "left"), ("assignment_statement", "left")],
    "java": [("variable_declarator", "name"), ("assignment_expression", "left")],
    "c": [("init_declarator", "declarator"), ("assignment_expression", "left")],
    "cpp": [("init_declarator", "declarator"), ("assignment_expression", "left")],
}


class DataFlowAnalyzerTS:
    def __init__(self, lang: str):
        self.lang = lang
        self.data_flows: list[tuple[str, str, str, int]] = []
        self.assignments: dict[str, list[str]] = {}
        self._cfg = LANG_CONFIG[lang]

    def analyze(self, tree, source: bytes) -> None:
        self._source = source
        self._walk(tree.root_node, class_stack=[], func_stack=[])

    def _walk(self, node, class_stack: list[str], func_stack: list[str]):
        cfg = self._cfg

        if node.type in cfg["class_types"]:
            name = node_ops.get_class_name(node, self.lang, self._source)
            new_class_stack = class_stack + [name] if name else class_stack
            for child in node.children:
                self._walk(child, new_class_stack, func_stack)
            return

        if node.type in cfg["function_types"]:
            fname = self._resolve_function_name(node)
            if fname:
                qname = qualified_name(class_stack, fname)
                self.assignments.setdefault(qname, [])
                for p in node_ops.get_function_params(node, self.lang, self._source):
                    self.assignments[qname].append(f"param:{p}")
                new_func_stack = func_stack + [qname]
            else:
                new_func_stack = func_stack
            body = node.child_by_field_name("body") or node
            for child in body.children:
                self._walk(child, class_stack, new_func_stack)
            return

        # assignments
        for assign_type, target_field in _ASSIGN_TARGET_FIELDS.get(self.lang, []):
            if node.type == assign_type:
                target = node.child_by_field_name(target_field)
                var_name = self._simple_var_name(target)
                if var_name and func_stack:
                    self.assignments.setdefault(func_stack[-1], []).append(var_name)

        # calls -> trace simple-identifier positional args
        if node.type in cfg["call_types"] and func_stack:
            callee = node_ops.get_call_callee(node, self.lang, self._source)
            args_node = node.child_by_field_name("arguments")
            if callee and args_node is not None:
                pos = 0
                simple_type = _SIMPLE_ID_TYPES[self.lang]
                for arg in args_node.children:
                    if arg.type in ("(", ")", ",", "argument_list"):
                        continue
                    if arg.type == simple_type:
                        self.data_flows.append(
                            (func_stack[-1], node_text(arg, self._source), callee, pos)
                        )
                    pos += 1

        for child in node.children:
            self._walk(child, class_stack, func_stack)

    def _resolve_function_name(self, node) -> str | None:
        name = node_ops.get_function_name(node, self.lang, self._source)
        if name:
            return name
        if self.lang in ("javascript", "typescript", "tsx") and node.parent is not None:
            parent = node.parent
            if parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node is not None:
                    return node_text(name_node, self._source)
        return None

    def _simple_var_name(self, target_node) -> str | None:
        if target_node is None:
            return None
        simple_type = _SIMPLE_ID_TYPES[self.lang]
        if target_node.type == simple_type:
            return node_text(target_node, self._source)
        # C/C++ init_declarator's "declarator" field can be nested under
        # pointer_declarator -- unwrap to the innermost identifier.
        if self.lang in ("c", "cpp"):
            from .common import first_identifier_text
            return first_identifier_text(target_node, self._source)
        return None
