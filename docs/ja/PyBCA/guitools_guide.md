---
title: GUI Tools Guide
parent: Japanese
nav_order: 15
---

# GUI Tools Guide

本ページは、PyBCA/BCL に付属する GUI ツール群の役割、依存関係、および利用方法を現行実装に基づいて整理したものである。

対象:

- PyBCA CellSpace Viewer
- BCL Editor
- Rule Editor

## 1. 依存

GUI 実行には `PySide6` が必要である。
加えて、用途に応じて以下の依存が必要となる。

- `numpy`
- `pyyaml`
- `torch`

`pyproject.toml` に GUI 専用依存は含まれていないため、手動で追加する必要がある。

## 2. PyBCA CellSpace Viewer

実体:

- `src/PyBCA/_legacy/guitools.py`
- `src/PyBCA/guitools.py` は互換ラッパ

起動:

```bash
python -m PyBCA._legacy.guitools
```

または:

```python
from PyBCA.guitools import main

main()
```

主な機能:

- CellSpace YAML の読み込みと保存
- Rule YAML の読み込み、追加、クリア
- Special Event ファイルの読み込み
- シミュレーション設定編集
- 1 step 実行
- 連続 run
- reset

注意:

- これはまだ legacy GUI 実装です
- `_load_default_files()` が参照する既定パスは、現行リポジトリ構成と一致しない場合があります
- 通常はファイルを手動ロードして使うのが安全です

## 3. BCL Editor

実装:

- `src/BCL/editor.py`

起動:

```bash
bcl-editor
```

このコマンドは `pyproject.toml` の `project.gui-scripts` で公開されている。

主な機能:

- `.bcl` の新規作成、読み込み、保存
- 点、矩形、線の配置
- `element` 定義の作成
- `element` の配置
- 選択範囲の copy/cut/paste
- YAML export
- Rule Editor の起動

自由度と制約:

- プリセットブラシは `0, 1, 2, -1, 3, 4` を提供する
- `Custom` 入力により `-128..127` の整数状態を直接配置できる
- キャンバスは動的に拡張されるため、初期サイズに縛られない
- BCL source pane は読み取り専用であり、完全なテキストエディタではない

主なショートカット:

- `Ctrl+N`: New
- `Ctrl+O`: Open
- `Ctrl+S`: Save
- `Ctrl+Shift+S`: Save As
- `Ctrl+E`: Export YAML
- `Ctrl+Z`: Undo
- `Ctrl+Shift+Z`: Redo
- `P`: Point tool
- `R`: Rectangle tool
- `L`: Line tool

## 4. Rule Editor

実装:

- `src/BCL/rule_editor.py`

起動:

```bash
python -m BCL.rule_editor
```

補足:

- 現状は `project.scripts` に登録されていません
- そのためモジュール実行が前提です

主な機能:

- Rule YAML の読み込みと保存
- prev/next パターン編集
- rule の追加と削除
- `probability` の直接編集
- `constants` alias を使った確率参照の維持

自由度と制約:

- pattern size は `3x3`, `5x5`, `7x7`, `9x9`
- カスタムブラシは `-128..127`
- `constants` と `conv` は読み込み・保存時に保持される
- ただし `constants` や `conv` を体系的に新規編集する専用 UI はない
- したがって local rule の試作には強いが、複雑な rule family 全体の著述では YAML 手編集を併用する方がよい

## 5. シミュレーション自由度と限界

GUI 群をシミュレーション実験の観点から整理すると、自由度は一様ではない。

- CellSpace の幾何配置と追加状態の埋め込み:
  `BCL Editor` が最も柔軟である
- local rule の `prev/next` 設計:
  `Rule Editor` が最も柔軟である
- 単一条件の step 実行確認:
  `CellSpace Viewer` が有効である

一方、次の点は GUI の守備範囲外、あるいは限定的である。

- `trial_constant_sweep` の本格的運用
- 多数 trial のバッチ実行
- `torchrun` による分散 trial 実行
- `constants` や `conv` を含む rule ファイル全体の体系的編集

特に CellSpace Viewer は legacy backend を用いており、GUI モードでは parallel trial が `1` 固定である。
従って、GUI は設計・可視確認の前段として有用であるが、研究用の本実験は `PyBCA.api.Engine` を用いたスクリプト実行へ移すのが適切である。

## 6. GUI と CLI/API の関係

- BCL Editor の YAML export は `BCLCompiler` を使います
- 生成した YAML はそのまま `Engine` の `cellspace_path` に渡せます
- Rule Editor で作った rule YAML も `Config.rule_paths` に渡せます
- Viewer で確認した CellSpace を API 実行に戻すこともできます

## 7. 現行実装における責務分離

- シミュレーション本体: `src/PyBCA/core`
- 公開実行 API: `src/PyBCA/api`
- GUI Viewer: `src/PyBCA/_legacy/guitools.py`
- BCL 編集 GUI: `src/BCL/editor.py`
- Rule 編集 GUI: `src/BCL/rule_editor.py`

GUI 全体が新 API 群へ全面移行したわけではない。
特に CellSpace Viewer は legacy 実装に依存するため、その前提のもとで利用する必要がある。
