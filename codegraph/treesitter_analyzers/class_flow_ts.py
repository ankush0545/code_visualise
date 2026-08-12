"""
treesitter_analyzers.class_flow_ts
------------------------------------
Multi-language equivalent of Flow_analyser.class_flow.ClassDiagramAnalyzer.
Same output shape:

    analyzer = ClassDiagramAnalyzerTS(lang)
    analyzer.analyze(tree, source_bytes)
    analyzer.classes   # dict[cls_name -> {"bases": [...], "attributes": [...], "methods": [...]}]

Go has no classes -- struct_type declarations are used as the nearest
analogue, with methods attached by receiver-type matching (a
method_declaration's receiver parameter type must textually match the
struct's type name). C structs get the same treatment as JS/Java/C++
classes but with empty bases (C has no inheritance) and no method
attachment (C has no methods-on-structs at the language level).
"""

from __future__ import annotations

from .common import iter_nodes, node_text
from .lang_config import LANG_CONFIG
from . import node_ops


class ClassDiagramAnalyzerTS:
    def __init__(self, lang: str):
        self.lang = lang
        self.classes: dict[str, dict] = {}

    def analyze(self, tree, source: bytes) -> None:
        cfg = LANG_CONFIG[self.lang]
        root = tree.root_node

        if self.lang == "go":
            self._analyze_go(root, source)
            return

        for node in iter_nodes(root):
            if node.type not in cfg["class_types"]:
                continue

            # C/C++ struct_specifier / class_specifier can appear as a
            # bare forward-declaration (no body field) -- skip those.
            if node.child_by_field_name("body") is None and self.lang not in (
                "javascript", "typescript", "tsx", "java",
            ):
                continue

            name = node_ops.get_class_name(node, self.lang, source)
            if not name:
                continue

            bases = node_ops.get_class_bases(node, self.lang, source)
            attrs = set(node_ops.get_class_attributes(node, self.lang, source))
            methods = []

            body = node.child_by_field_name("body")
            if body is not None:
                for child in body.children:
                    if child.type in LANG_CONFIG[self.lang]["function_types"]:
                        fname = node_ops.get_function_name(child, self.lang, source)
                        if not fname:
                            continue
                        params = node_ops.get_function_params(child, self.lang, source)
                        methods.append(f"{fname}({', '.join(params)})")
                        attrs |= self._self_attrs_in_method(child, source)

            self.classes[name] = {
                "bases": bases,
                "attributes": sorted(attrs),
                "methods": methods,
            }

    # -----------------------------------------------------------------
    # Go: struct_type is the closest analogue to a class. Methods are
    # declared separately (func (s *Server) Method(...)) so we do a
    # second pass to attach them by matching the receiver's type name.
    # -----------------------------------------------------------------
    def _analyze_go(self, root, source: bytes) -> None:
        structs: dict[str, dict] = {}
        for node in iter_nodes(root):
            if node.type != "type_declaration":
                continue
            for spec in node.children:
                if spec.type != "type_spec":
                    continue
                type_field = spec.child_by_field_name("type")
                if type_field is None or type_field.type != "struct_type":
                    continue
                name_node = spec.child_by_field_name("name")
                if name_node is None:
                    continue
                name = node_text(name_node, source)
                attrs = node_ops.get_class_attributes(type_field, "go", source)
                structs[name] = {"bases": [], "attributes": attrs, "methods": []}

        for node in iter_nodes(root):
            if node.type != "method_declaration":
                continue
            receiver = node.child_by_field_name("receiver")
            if receiver is None:
                continue
            recv_type = None
            recv_var = None
            for n in iter_nodes(receiver):
                if n.type == "type_identifier":
                    recv_type = node_text(n, source)
                if n.type == "identifier" and recv_var is None:
                    recv_var = node_text(n, source)
            if recv_type not in structs:
                continue
            fname = node_ops.get_function_name(node, "go", source)
            if not fname:
                continue
            params = node_ops.get_function_params(node, "go", source)
            structs[recv_type]["methods"].append(f"{fname}({', '.join(params)})")

            # capture recvVar.field = ... as an attribute, same spirit
            # as self.x = ... in the Python analyzer
            if recv_var:
                body = node.child_by_field_name("body")
                if body is not None:
                    for n in iter_nodes(body):
                        if n.type == "assignment_statement":
                            left = n.children[0] if n.children else None
                            self._collect_selector_attr(left, recv_var, source, structs[recv_type]["attributes"])

        self.classes = structs

    @staticmethod
    def _collect_selector_attr(node, recv_var: str, source: bytes, attrs_out: list[str]):
        if node is None:
            return
        for n in iter_nodes(node):
            if n.type == "selector_expression":
                obj = n.child_by_field_name("operand")
                field = n.child_by_field_name("field")
                if obj is not None and field is not None and node_text(obj, source) == recv_var:
                    fname = node_text(field, source)
                    if fname not in attrs_out:
                        attrs_out.append(fname)

    # -----------------------------------------------------------------
    # self/this attribute assignments inside a method body -- mirrors
    # the Python analyzer's `self.x = ...` -> attributes capture.
    # -----------------------------------------------------------------
    def _self_attrs_in_method(self, func_node, source: bytes) -> set[str]:
        attrs: set[str] = set()
        self_names = {
            "javascript": "this", "typescript": "this", "tsx": "this", "java": "this",
        }
        member_field = {
            "javascript": ("member_expression", "object", "property"),
            "typescript": ("member_expression", "object", "property"),
            "tsx": ("member_expression", "object", "property"),
            "java": ("field_access", "object", "field"),
            "cpp": ("field_expression", "argument", "field"),
        }
        self_name = self_names.get(self.lang)
        member_info = member_field.get(self.lang)
        if not member_info:
            return attrs
        member_type, obj_field, prop_field = member_info

        assign_types = {"assignment_expression"} if self.lang != "java" else {"assignment_expression"}
        for n in iter_nodes(func_node):
            if n.type not in assign_types:
                continue
            left = n.child_by_field_name("left")
            if left is None or left.type != member_type:
                continue
            obj = left.child_by_field_name(obj_field)
            prop = left.child_by_field_name(prop_field)
            if prop is None:
                continue
            if self.lang == "cpp":
                # C++: only count bare `this->x = ...`, obj must be a
                # `this` expression node
                if obj is None or node_text(obj, source) != "this":
                    continue
            elif self_name and (obj is None or node_text(obj, source) != self_name):
                continue
            attrs.add(node_text(prop, source))
        return attrs
