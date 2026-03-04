---
title: BCL Guide (v0.1)
nav_order: 10
---

# Brownian Circuits Language (BCL)

このページは `src/BCL/compiler.py` の現行実装 (`BCLCompiler`) に厳密に合わせた仕様です。
記法は「将来仕様」ではなく、**今コンパイルできる構文**だけを対象にしています。

## 1. できること

BCL はセル配置を記述し、YAML（`[{coord:{x,y}, value:v}, ...]`）へ変換します。

- 基本セル配置: `place.signal_line`, `place.token`, `place.recycle_bin`, `place.cell`
- 座標シンボル: `coord.define(name, x, y)`
- 要素マクロ: `element Name(param) { ... }` と `place.Name(inst, param[ex, ey])`

## 2. コメント

- `#` 以降は行末までコメントとして無視されます。

## 3. 文法（実装準拠）

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

`EXPR` は `BCLCompiler._eval_value` に従います。未定義シンボルや不正式は `SyntaxError` になります。

## 4. 値の意味

- `place.signal_line(x, y)` -> `value = 1`
- `place.token(x, y)` -> `value = 2`
- `place.recycle_bin(x, y)` -> `value = -1`
- `place.cell(x, y, v)` -> `value = v`（式評価後の整数）

## 5. element 展開ルール

`element` は前処理で展開されます。

- 定義: `element Join_Fork_right(io_a){ ... }`
- 呼び出し: `place.Join_Fork_right(inst1, io_a[20, -5])`

実装上の重要点:

1. `element` のパラメータは現実装では 1 つだけ。
2. 呼び出し側の `param[...]` の `param` 名は、定義側パラメータ名と一致必須。
3. `place.Element(...)` は前処理で `coord.define(...)` と `place.*` の通常行に展開されます。

## 6. 非対応（現実装）

以下は現行コンパイラでは解釈しません（`Unsupported line` などで失敗）:

- `input: ...`, `output: ...`
- `construct info(...) { ... }`
- 複数引数 element パラメータ
- 任意の演算式（`a.x + b.x` など）

## 7. CLI

`pyproject.toml` の script で `bcl` が使えます。

```bash
bcl INPUT.bcl -o OUTPUT.yaml
```

補足:

- `src/BCL/cli.py` のヘルプ文には `npz` とありますが、`BCLCompiler.write_yaml` は実際には YAML を書きます。
- 出力のトップレベルはリストです。

## 8. Python API

```python
from BCL.compiler import BCLCompiler

comp = BCLCompiler()
comp.read_file("Sample/bclfile/sample.bcl")
comp.parse()
ir = comp.lower_to_ir()
comp.write_yaml(ir, "/tmp/cellspace.yaml")
```

## 9. 最小例

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

## 10. BCL Editor との関係

`bcl-editor`（`src/BCL/editor.py`）はこの文法を内部で `BCLCompiler` に渡して検証・YAML出力します。
GUIの詳細は [PyBCA GUI Tools Guide](../../PyBCA/guitools_guide.md) を参照してください。
