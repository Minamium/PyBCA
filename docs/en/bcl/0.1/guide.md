---
title: BCL Guide
parent: English
nav_order: 15
---

# Brownian Circuits Language (BCL)

This page is aligned strictly with the current implementation of `BCLCompiler` in `src/BCL/compiler.py`.
It documents only the syntax that is accepted now.

Related pages:

- [PyBCA Guide](../../PyBCA/PyBCA_guide.md)
- [GUI Tools Guide](../../PyBCA/guitools_guide.md)

## 1. Purpose Of BCL

BCL is a text format for describing CellSpace layouts and compiling them into YAML.
The generated YAML can be passed directly to `Config.cellspace_path`.

The top-level YAML output is a list. Each entry looks like this:

```yaml
- coord:
    x: 10
    y: 20
  value: 1
```

## 2. Supported Features

- `coord.define(name, x, y)`
- `place.signal_line(x, y)`
- `place.token(x, y)`
- `place.recycle_bin(x, y)`
- `place.cell(x, y, value)`
- `element Name(param) { ... }`
- `place.Name(instance, param[expr, expr])`

Comments start with `#` and continue to the end of the line.

## 3. Grammar

```text
program            := { statement | comment | blank }

statement          := coord_define
                   | place_signal_line
                   | place_token
                   | place_recycle_bin
                   | place_cell
                   | element_define
                   | place_element

coord_define       := "coord.define(" IDENT "," EXPR "," EXPR ")"
place_signal_line  := "place.signal_line(" EXPR "," EXPR ")"
place_token        := "place.token(" EXPR "," EXPR ")"
place_recycle_bin  := "place.recycle_bin(" EXPR "," EXPR ")"
place_cell         := "place.cell(" EXPR "," EXPR "," EXPR ")"

element_define     := "element" IDENT "(" IDENT ")" "{" { statement_like_line } "}"
place_element      := "place." IDENT "(" IDENT "," IDENT "[" EXPR "," EXPR "]" ")"

EXPR               := INT
                   | IDENT ".x" [ ("+"|"-") INT ]
                   | IDENT ".y" [ ("+"|"-") INT ]

IDENT              := [A-Za-z_][A-Za-z0-9_]*
INT                := -?[0-9]+
```

Actual expression handling follows `BCLCompiler._eval_value()`.

## 4. Cell Values

- `place.signal_line(x, y)` -> `value = 1`
- `place.token(x, y)` -> `value = 2`
- `place.recycle_bin(x, y)` -> `value = -1`
- `place.cell(x, y, v)` -> `value = v`

## 5. `element` Expansion

`element` blocks are expanded in preprocessing.

Example:

```bcl
element Dot(p){
    place.cell(p.x, p.y, 3)
}

place.Dot(dot_1, p[15, 25])
```

Implementation constraints:

1. only one parameter is supported
2. the `param[...]` name at the call site must match the definition parameter
3. multi-parameter `element` definitions are not supported

## 6. Unsupported Syntax

The current compiler rejects:

- `input: ...`
- `output: ...`
- `construct info(...) { ... }`
- multi-parameter `element`
- arbitrary expressions such as `a.x + b.x`

## 7. CLI

The `bcl` command is exposed through `project.scripts` in `pyproject.toml`.

```bash
bcl INPUT.bcl -o OUTPUT.yaml
```

Example:

```bash
bcl Sample/bclfile/sample.bcl -o /tmp/sample.yaml
```

## 8. Python API

```python
from BCL.compiler import BCLCompiler

comp = BCLCompiler()
comp.read_file("Sample/bclfile/sample.bcl")
comp.parse()
ir = comp.lower_to_ir()
comp.write_yaml(ir, "/tmp/cellspace.yaml")
```

For the one-shot path:

```python
from BCL.compiler import BCLCompiler

BCLCompiler().compile_file("Sample/bclfile/sample.bcl", "/tmp/sample.yaml")
```

## 9. Minimal Example

```bcl
# anchor
coord.define(io_a, 10, 20)

# primitive placements
place.signal_line(io_a.x, io_a.y)
place.token(io_a.x+1, io_a.y)
place.recycle_bin(io_a.x+2, io_a.y)

# macro
element Dot(p){
    place.cell(p.x, p.y, 3)
}

place.Dot(dot_1, p[15, 25])
```

## 10. From BCL To PyBCA

1. compile `.bcl` into YAML
2. pass the generated YAML into `Config.cellspace_path`
3. pass Rule YAML files into `Config.rule_paths`
4. run `Engine.run()`

## 11. Consistency With Samples

At audit time, the following were confirmed.

- all `Sample/bclfile/*.bcl` files parsed successfully
- `BNN.bcl` matched `Sample/Cellspace/BNN.yaml`
- `C-Join_from_JF.bcl` matched `Sample/Cellspace/2-way_C-Join.yaml`

## 12. Relationship To The GUI

- `bcl-editor` uses `BCLCompiler` internally for validation and YAML export
- `Rule Editor` operates on Rule YAML files
- see [GUI Tools Guide](../../PyBCA/guitools_guide.md) for GUI details
