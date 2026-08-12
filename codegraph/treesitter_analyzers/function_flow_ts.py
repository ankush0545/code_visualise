"""
treesitter_analyzers.function_flow_ts
----------------------------------------
Multi-language equivalent of Flow_analyser.function_flow.CallFlowAnalyzer.
Same output shape:

    analyzer = CallFlowAnalyzerTS(lang)
    analyzer.analyze(tree, source_bytes)
    analyzer.calls   # list[(caller: str, callee: str)]

Calls made directly inside a function body are attributed to that
function; calls inside a *nested* function/arrow-function defined
within it are attributed to the nested one instead -- same rule as the
Python CallFlowAnalyzer ("not inside nested functions... those get
attributed to themselves").
"""

from __future__ import annotations

from .common import qualified_name, node_text
from .lang_config import LANG_CONFIG
from . import node_ops


class CallFlowAnalyzerTS:
    def __init__(self, lang: str):
        self.lang = lang
        self.calls: list[tuple[str, str]] = []
        # every qualified function name seen, even with zero calls made --
        # needed so extract_repo.py can still emit a function_call_flow
        # entry for functions that call nothing (they're still Function
        # nodes in graph.py's build_unified_graph).
        self.function_names: list[str] = []
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
                self.function_names.append(qname)
                new_func_stack = func_stack + [qname]
            else:
                new_func_stack = func_stack
            body = node.child_by_field_name("body") or node
            for child in body.children:
                self._walk(child, class_stack, new_func_stack)
            return

        if node.type in cfg["call_types"] and func_stack:
            callee = node_ops.get_call_callee(node, self.lang, self._source)
            if callee:
                self.calls.append((func_stack[-1], callee))
            # still descend -- calls can nest, e.g. f(g(x))

        for child in node.children:
            self._walk(child, class_stack, func_stack)

    def _resolve_function_name(self, node) -> str | None:
        """JS/TS arrow functions and function expressions assigned to a
        `const name = ...` don't carry their own name field -- pull it
        from the enclosing variable_declarator instead."""
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
