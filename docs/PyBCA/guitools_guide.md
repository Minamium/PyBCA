---
title: GUI Tools Guide (PyBCA/BCL)
nav_order: 30
---

# GUI Tools Guide

このページは現行実装の GUI ツールを整理したものです。

- PyBCA CellSpace Viewer（legacy 実装を互換ラップ）
- BCL Editor
- Rule Editor

## 1. 依存

GUI 実行には `PySide6` が必要です。加えて `numpy`, `pyyaml`、シミュレーション用途では `torch` が必要です。

## 2. PyBCA CellSpace Viewer

実体は `src/PyBCA/_legacy/guitools.py` で、`src/PyBCA/guitools.py` は互換ラッパです。

### 起動

```bash
python -m PyBCA._legacy.guitools
```

または:

```python
from PyBCA.guitools import main
main()
```

### 主な機能

- CellSpace YAML の読み込み/保存
- Rule YAML の読み込み・追加・クリア・表示
- Special Event ファイルの読み込みとオーバーレイ表示
- シミュレーション設定 (`global_prob`, `seed`, `device`, `state_gate_*`)
- 1 step 実行 / 連続実行 / リセット

### 注意

`_load_default_files()` のデフォルトパスは現行リポジトリ構成では存在しないことが多いため、通常は手動ロード前提です。

## 3. BCL Editor

実装: `src/BCL/editor.py`。
`pyproject.toml` で GUI スクリプト `bcl-editor` が定義されています。

### 起動

```bash
bcl-editor
```

### 主な機能

- `.bcl` 読み込み/保存
- キャンバス上のポイント・矩形・ライン編集
- element 定義の作成とドラッグ配置
- 選択範囲 Copy/Cut/Paste/Move
- YAML エクスポート（内部で `BCLCompiler` を使用）
- Rule Editor 起動

### 主なショートカット

- `Ctrl+N`: New
- `Ctrl+O`: Open BCL
- `Ctrl+S`: Save
- `Ctrl+Shift+S`: Save As
- `Ctrl+E`: Export YAML
- `Ctrl+Z`: Undo
- `Ctrl+Shift+Z`: Redo
- `P` / `R` / `L`: Point / Rectangle / Line ツール
- `Ctrl+C` / `Ctrl+X` / `Ctrl+V`: Copy/Cut/Paste

## 4. Rule Editor

実装: `src/BCL/rule_editor.py`。
現状 `project.scripts` には未登録のため、モジュール実行で起動します。

### 起動

```bash
python -m BCL.rule_editor
```

### 主な機能

- Rule YAML の読み込み/保存
- prev/next パターン編集
- rule 追加/削除
- `probability` 直接値編集
- `constants` による alias 参照（`probability: *alias`）の維持

## 5. 実装整合のポイント

- BCL Editor と CLI コンパイラは同じ `BCLCompiler` 文法に依存
- PyBCA Viewer は現時点で legacy GUI 実装を使用
- GUI で保存した BCL/YAML は、CLI/API からそのまま再利用可能
