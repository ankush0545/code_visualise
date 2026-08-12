"""
treesitter_analyzers.lang_config
----------------------------------
Per-language node-type tables, confirmed empirically against each
grammar (see the dump-the-parse-tree commands used to build this file --
these are NOT guessed from docs, every type here was verified against
tree_sitter_languages' actual grammars for js/ts/go/java/c/cpp).

Node type sets only -- the actual extraction logic (which field holds a
name, how a callee is resolved, how a class's base list is read) lives
in node_ops.py because those differ too much per-language to flatten
into a single generic path.
"""

LANG_CONFIG = {
    "javascript": dict(
        function_types={
            "function_declaration", "method_definition",
            "function_expression", "arrow_function",
            "generator_function_declaration",
        },
        class_types={"class_declaration", "class"},
        call_types={"call_expression", "new_expression"},
        import_types={"import_statement"},
        require_call_names={"require"},  # CommonJS: const x = require('y')
    ),
    "typescript": dict(
        function_types={
            "function_declaration", "method_definition",
            "function_expression", "arrow_function",
            "generator_function_declaration",
        },
        class_types={"class_declaration", "class"},
        call_types={"call_expression", "new_expression"},
        import_types={"import_statement"},
        require_call_names={"require"},
    ),
    "tsx": dict(
        function_types={
            "function_declaration", "method_definition",
            "function_expression", "arrow_function",
            "generator_function_declaration",
        },
        class_types={"class_declaration", "class"},
        call_types={"call_expression", "new_expression"},
        import_types={"import_statement"},
        require_call_names={"require"},
    ),
    "go": dict(
        function_types={"function_declaration", "method_declaration"},
        class_types={"type_declaration"},  # filtered to struct_type in node_ops
        call_types={"call_expression"},
        import_types={"import_declaration"},
    ),
    "java": dict(
        function_types={"method_declaration", "constructor_declaration"},
        class_types={"class_declaration", "interface_declaration"},
        call_types={"method_invocation", "object_creation_expression"},
        import_types={"import_declaration"},
    ),
    "c": dict(
        function_types={"function_definition"},
        class_types={"struct_specifier"},
        call_types={"call_expression"},
        import_types={"preproc_include"},
    ),
    "cpp": dict(
        function_types={"function_definition"},
        class_types={"class_specifier", "struct_specifier"},
        call_types={"call_expression"},
        import_types={"preproc_include"},
    ),
}
