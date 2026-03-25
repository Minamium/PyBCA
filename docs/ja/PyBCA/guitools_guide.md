---
title: GUI Tools Guide
parent: Japanese
nav_order: 14
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

## 5. GUI と CLI/API の関係

- BCL Editor の YAML export は `BCLCompiler` を使います
- 生成した YAML はそのまま `Engine` の `cellspace_path` に渡せます
- Rule Editor で作った rule YAML も `Config.rule_paths` に渡せます
- Viewer で確認した CellSpace を API 実行に戻すこともできます

## 6. 現行実装における責務分離

- シミュレーション本体: `src/PyBCA/core`
- 公開実行 API: `src/PyBCA/api`
- GUI Viewer: `src/PyBCA/_legacy/guitools.py`
- BCL 編集 GUI: `src/BCL/editor.py`
- Rule 編集 GUI: `src/BCL/rule_editor.py`

GUI 全体が新 API 群へ全面移行したわけではない。
特に CellSpace Viewer は legacy 実装に依存するため、その前提のもとで利用する必要がある。
