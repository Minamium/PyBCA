---
title: BCL Guide
parent: Japanese
nav_order: 15
---

# Brownian Circuits Language (BCL)

本ページは `src/BCL/compiler.py` に実装された `BCLCompiler` の受理文法に基づき、現行 BCL の仕様を記述する。
対象は将来構想ではなく、現時点で実際にコンパイル可能な構文に限る。

関連:

- [PyBCA Guide](../../PyBCA/PyBCA_guide.md)
- [GUI Tools Guide](../../PyBCA/guitools_guide.md)

## 1. BCL の役割

BCL は CellSpace をテキストで記述し、YAML へ変換するための言語である。
コンパイル後の YAML は、そのまま `Config.cellspace_path` に渡すことができる。

出力 YAML のトップレベルはリストで、各要素は次の形です。

```yaml
- coord:
    x: 10
    y: 20
  value: 1
```

## 2. 基本機能

- `coord.define(name, x, y)`
- `place.signal_line(x, y)`
- `place.token(x, y)`
- `place.recycle_bin(x, y)`
- `place.cell(x, y, value)`
- `element Name(param) { ... }`
- `place.Name(instance, param[expr, expr])`

コメントは `#` 以降が行末まで無視される。

## 3. 文法

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

`EXPR` の評価は `BCLCompiler._eval_value()` に従う。

## 4. 値の意味

- `place.signal_line(x, y)` -> `value = 1`
- `place.token(x, y)` -> `value = 2`
- `place.recycle_bin(x, y)` -> `value = -1`
- `place.cell(x, y, v)` -> `value = v`

## 5. `element` 展開

`element` は前処理で通常行へ展開される。

例:

```bcl
element Dot(p){
    place.cell(p.x, p.y, 3)
}

place.Dot(dot_1, p[15, 25])
```

実装上の制約:

1. パラメータは 1 個だけ
2. 呼び出し側の `param[...]` の `param` 名は定義側と一致必須
3. 複数引数 `element` は未対応

## 6. 非対応構文

以下は現行 compiler では受理されない。

- `input: ...`
- `output: ...`
- `construct info(...) { ... }`
- 複数引数 `element`
- `a.x + b.x` のような任意式

## 7. CLI

`pyproject.toml` の `project.scripts` により `bcl` コマンドが公開されています。

```bash
bcl INPUT.bcl -o OUTPUT.yaml
```

例:

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

まとめて実行するなら:

```python
from BCL.compiler import BCLCompiler

BCLCompiler().compile_file("Sample/bclfile/sample.bcl", "/tmp/sample.yaml")
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

## 10. BCL から PyBCA へ

1. `.bcl` を YAML に変換
2. 生成 YAML を `Config.cellspace_path` に設定
3. Rule YAML を `Config.rule_paths` に設定
4. `Engine.run()` で実行

## 11. 実装準拠性の確認

現行の sample 資産について、少なくとも以下の整合性を確認している。

- `Sample/bclfile/*.bcl` は parse 成功
- `BNN.bcl` は `Sample/Cellspace/BNN.yaml` と一致
- `C-Join_from_JF.bcl` は `Sample/Cellspace/2-way_C-Join.yaml` と一致

## 12. GUI との関係

- `bcl-editor` は内部で `BCLCompiler` を使って検証と YAML export を行います
- `Rule Editor` は rule YAML 側を編集します
- GUI 詳細は [GUI Tools Guide](../../PyBCA/guitools_guide.md) を参照してください
