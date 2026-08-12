"""
treesitter_analyzers.node_ops
-------------------------------
Per-language "how do I get X out of this node" functions. This is the
part that genuinely can't be flattened into one generic path across
6 languages -- field names and structure differ too much. Each function
below is dispatched on `lang` (one of: javascript, typescript, tsx, go,
java, c, cpp).

Known, deliberate simplifications (documented rather than hidden):
  - JS/TS arrow functions assigned to a const get their name from the
    enclosing variable_declarator, resolved one level up in
    class_flow_ts.py/function_flow_ts.py, not here.
  - Go has no classes; struct_type is treated as the closest analogue,
    with methods attached by matching a method_declaration's receiver
    type name against the struct's type name (see class_flow_ts.py).
  - C has no classes; struct_specifier is exposed the same way, with
    empty bases/methods (C has no method-on-struct concept at the
    language level -- callers pass the struct explicitly).
  - Callee names are normalized to their LAST segment (mirrors
    normalize_call_name() in your graph.py), e.g. `bar.baz()` -> "baz",
    `p->name` access isn't a call so doesn't apply, `fmt.Println()` ->
    "Println". This intentionally loses which object/module the call
    is against -- same tradeoff your existing Python pipeline makes.
"""

from __future__ import annotations

from .common import node_text, iter_nodes, first_identifier_text

# ---------------------------------------------------------------------------
# Function name + params
# ---------------------------------------------------------------------------

def get_function_name(node, lang: str, source: bytes) -> str | None:
    if lang in ("javascript", "typescript", "tsx"):
        name_node = node.child_by_field_name("name")
        return node_text(name_node, source) if name_node else None

    if lang == "go":
        name_node = node.child_by_field_name("name")
        return node_text(name_node, source) if name_node else None

    if lang == "java":
        name_node = node.child_by_field_name("name")
        return node_text(name_node, source) if name_node else None

    if lang in ("c", "cpp"):
        declarator = node.child_by_field_name("declarator")
        # declarator chain: pointer_declarator* -> function_declarator -> identifier/field_identifier
        return first_identifier_text(declarator, source)

    return None


def get_function_params(node, lang: str, source: bytes) -> list[str]:
    if lang in ("javascript", "typescript", "tsx"):
        params_node = node.child_by_field_name("parameters")
        return _js_param_names(params_node, source) if params_node else []

    if lang == "go":
        params_node = node.child_by_field_name("parameters")
        return _go_param_names(params_node, source) if params_node else []

    if lang == "java":
        params_node = node.child_by_field_name("parameters")
        return _java_param_names(params_node, source) if params_node else []

    if lang in ("c", "cpp"):
        declarator = node.child_by_field_name("declarator")
        params_node = None
        for n in iter_nodes(declarator) if declarator else []:
            if n.type == "parameter_list":
                params_node = n
                break
        return _c_param_names(params_node, source) if params_node else []

    return []


def _js_param_names(params_node, source: bytes) -> list[str]:
    names = []
    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue
        # identifier | required_parameter | optional_parameter |
        # object_pattern (destructured) | assignment_pattern (default value)
        name = first_identifier_text(child, source)
        if name:
            names.append(name)
    return names


def _go_param_names(params_node, source: bytes) -> list[str]:
    names = []
    for child in params_node.children:
        if child.type != "parameter_declaration":
            continue
        for sub in child.children:
            if sub.type == "identifier":
                names.append(node_text(sub, source))
    return names


def _java_param_names(params_node, source: bytes) -> list[str]:
    names = []
    for child in params_node.children:
        if child.type != "formal_parameter":
            continue
        name_node = child.child_by_field_name("name")
        if name_node:
            names.append(node_text(name_node, source))
    return names


def _c_param_names(params_node, source: bytes) -> list[str]:
    names = []
    for child in params_node.children:
        if child.type != "parameter_declaration":
            continue
        declarator = child.child_by_field_name("declarator")
        name = first_identifier_text(declarator, source) if declarator else None
        if name:
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Class / struct name + bases + attributes
# ---------------------------------------------------------------------------

def get_class_name(node, lang: str, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    return node_text(name_node, source) if name_node else None


def get_class_bases(node, lang: str, source: bytes) -> list[str]:
    if lang in ("javascript", "typescript", "tsx"):
        heritage = None
        for child in node.children:
            if child.type == "class_heritage":
                heritage = child
                break
        if not heritage:
            return []
        return [node_text(n, source) for n in heritage.children
                if n.type in ("identifier", "type_identifier", "member_expression")]

    if lang == "java":
        bases = []
        superclass = node.child_by_field_name("superclass")
        if superclass:
            for n in iter_nodes(superclass):
                if n.type == "type_identifier":
                    bases.append(node_text(n, source))
                    break
        interfaces = node.child_by_field_name("interfaces")
        if interfaces:
            bases.extend(node_text(n, source) for n in iter_nodes(interfaces)
                         if n.type == "type_identifier")
        return bases

    if lang == "cpp":
        bases = []
        for child in node.children:
            if child.type == "base_class_clause":
                bases.extend(node_text(n, source) for n in child.children
                             if n.type == "type_identifier")
        return bases

    return []  # go, c: no inheritance concept


def get_class_attributes(node, lang: str, source: bytes) -> list[str]:
    """Field-level attributes declared directly on the class/struct body
    (NOT self.x = assignments inside methods -- those are collected
    separately in class_flow_ts.py, mirroring your Python
    ClassDiagramAnalyzer's two attribute sources)."""
    attrs: set[str] = set()

    if lang in ("javascript", "typescript", "tsx"):
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type in ("field_definition", "public_field_definition"):
                    name_node = child.child_by_field_name("property") or child.child_by_field_name("name")
                    if name_node:
                        attrs.add(node_text(name_node, source))

    elif lang == "java":
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "field_declaration":
                    for n in iter_nodes(child):
                        if n.type == "variable_declarator":
                            name_node = n.child_by_field_name("name")
                            if name_node:
                                attrs.add(node_text(name_node, source))

    elif lang == "go":
        # node here is the struct_type, not type_declaration -- see class_flow_ts.py
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "field_declaration":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        attrs.add(node_text(name_node, source))

    elif lang in ("c", "cpp"):
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "field_declaration":
                    name = first_identifier_text(
                        child.child_by_field_name("declarator"), source
                    )
                    if name:
                        attrs.add(name)

    return sorted(attrs)


# ---------------------------------------------------------------------------
# Call target resolution -- always normalized to the LAST segment, same
# convention as graph.py's normalize_call_name() for the Python side.
# ---------------------------------------------------------------------------

def get_call_callee(node, lang: str, source: bytes) -> str | None:
    if lang in ("javascript", "typescript", "tsx"):
        func_node = node.child_by_field_name("function") or node.child_by_field_name("constructor")
        return _last_segment(func_node, source, member_types=("member_expression",), prop_field="property")

    if lang == "go":
        func_node = node.child_by_field_name("function")
        return _last_segment(func_node, source, member_types=("selector_expression",), prop_field="field")

    if lang == "java":
        if node.type == "method_invocation":
            name_node = node.child_by_field_name("name")
            return node_text(name_node, source) if name_node else None
        if node.type == "object_creation_expression":
            type_node = node.child_by_field_name("type")
            return node_text(type_node, source) if type_node else None
        return None

    if lang in ("c", "cpp"):
        func_node = node.child_by_field_name("function")
        return _last_segment(
            func_node, source,
            member_types=("field_expression", "qualified_identifier"),
            prop_field="field",
        )

    return None


def _last_segment(node, source: bytes, member_types: tuple[str, ...], prop_field: str) -> str | None:
    if node is None:
        return None
    if node.type in member_types:
        prop = node.child_by_field_name(prop_field) or node.child_by_field_name("name")
        if prop:
            return node_text(prop, source)
        return first_identifier_text(node, source)
    if node.type in ("identifier", "field_identifier", "type_identifier"):
        return node_text(node, source)
    return first_identifier_text(node, source)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

def get_import_names(node, lang: str, source: bytes) -> list[str]:
    """Returns the raw module/package identifiers referenced by one
    import statement (a single JS import can reference one path; a
    single Go import_declaration node can wrap several import_specs)."""
    if lang in ("javascript", "typescript", "tsx"):
        for n in iter_nodes(node):
            if n.type == "string":
                text = node_text(n, source).strip("'\"")
                return [text] if text else []
        return []

    if lang == "go":
        names = []
        for n in iter_nodes(node):
            if n.type in ("interpreted_string_literal", "raw_string_literal"):
                text = node_text(n, source).strip('"').strip("`")
                if text:
                    names.append(text)
        return names

    if lang == "java":
        for n in iter_nodes(node):
            if n.type == "scoped_identifier":
                return [node_text(n, source)]
        return []

    if lang in ("c", "cpp"):
        for n in iter_nodes(node):
            if n.type in ("string_literal", "system_lib_string"):
                return [node_text(n, source).strip('"<>')]
        return []

    return []
