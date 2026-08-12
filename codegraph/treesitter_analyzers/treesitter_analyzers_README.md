# treesitter_analyzers

Tree-sitter based equivalents of your `Flow_analyser` package
(`class_flow.py`, `function_flow.py`, `data_flow.py`), covering
**JavaScript, TypeScript (+TSX), Go, Java, C, and C++**.

Output schema is identical to what `graph.py`'s `build_unified_graph()`
already expects, so **no changes to graph.py, Label_extraction.py,
edge_label_extraction.py, or model.py are needed.** This is a drop-in
additional source feeding the same pipeline your Python/AST extractor
already writes into `Data/output_*.json`.

## Install

```bash
pip install "tree_sitter==0.21.3" tree_sitter_languages --break-system-packages
```

`tree_sitter==0.21.3` is pinned deliberately — `tree_sitter_languages`'s
prebuilt grammars use the pre-0.22 C API. Installing a newer
`tree_sitter` (e.g. the current 0.26) will raise
`TypeError: __init__() takes exactly 1 argument (2 given)` at parse
time even though import succeeds cleanly.

## Usage

```python
from treesitter_analyzers.extract_repo import extract_repo
import json

raw = extract_repo("./repo_cache/some-js-repo",
                    repo_url="https://github.com/facebook/react")

# `raw` has the exact same 5 keys your Data/output_<id>.json files do:
#   file_index, class_diagram, function_call_flow,
#   variables_per_function, data_flow
with open("Data/output_react.json", "w") as f:
    json.dump(raw, f)

# from here on, main.py's existing pipeline handles it unmodified:
#   merge_raw_analysis -> build_unified_graph -> build_labeled_data
#   -> build_edge_labels -> modelling_multitask
```

For a mixed-language repo (say, a monorepo with Python backend + a
JS/TS frontend), run BOTH extractors and merge their dicts with your
existing `merge_raw_analysis([path_a, path_b])` before calling
`build_unified_graph()` — since `graph.py`'s `file_id()` keys nodes by
`(repo_url, filepath)`, files from different languages never collide.

## What's covered per language

| | function defs | class/struct defs | calls | imports | data flow |
|---|---|---|---|---|---|
| JS/TS/TSX | function decl, method, arrow fn (via `const x = ...`), generator | class + `extends` | call + `new` | `import`, bare `require()` args not yet traced | assignments + positional-identifier call args |
| Go | func + method (receiver-matched to struct) | `struct` (closest analogue; no inheritance) | call | `import` | short/regular var decls + call args |
| Java | method + constructor | class + interface (`extends`/`implements`) | method invocation + `new` | `import` | var decls + call args |
| C | function definition | `struct` (no methods-on-struct, no inheritance) | call | `#include` | declarations + call args |
| C++ | function/method definition | `class`/`struct` + public/private inheritance | call | `#include` | declarations + call args |

## Known, deliberate limitations (see docstrings for detail)

- Only **simple identifier** arguments are traced for data flow (`f(x)`
  traces `x`; `f(x + 1)` or `f(get_x())` doesn't) — matches the scope
  of your existing Python `DataFlowAnalyzer`.
- Call targets are normalized to their **last segment**
  (`bar.baz()` → `"baz"`) — matches `graph.py`'s `normalize_call_name()`
  convention for the Python side, with the same tradeoff: it can't
  tell `foo.helper()` from `bar.helper()` and will link both to any
  function literally named `helper` anywhere in the repo (same-repo,
  same-name resolution is `graph.py`'s behavior, not something this
  package changes — but it's worth knowing this now applies **across
  languages too**: a JS `helper()` and a C `helper()` in the same repo
  will get silently linked as the same call target).
- `IGNORE_DIRS` in `extract_repo.py` skips `node_modules`, `vendor`,
  `dist`, `build`, `target`, `.venv`, etc. — extend that set if your
  repos vendor dependencies somewhere else.
- CommonJS `const x = require('y')` is recognized as an import
  statement's *sibling* pattern is not yet wired into `class_diagram`/
  `function_call_flow` the way ES `import` is — only `import_statement`
  nodes are parsed for now. Flag if you need `require()` tracked too.

## Verified against

Tested end-to-end against a synthetic 5-file repo (JS class w/
inheritance + arrow-free methods, Go struct w/ receiver method, Java
class w/ inheritance, C++ class w/ `this->`, plain C function), fed
directly into your unmodified `graph.py`. Produced correct File → Class
→ Function nodes and DEFINES/HAS_METHOD/INHERITS/CALLS/FLOWS_TO edges,
including a cross-file call resolving correctly to the right Function
node.
