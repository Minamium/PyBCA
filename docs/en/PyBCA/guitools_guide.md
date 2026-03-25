---
title: GUI Tools Guide
parent: English
nav_order: 14
---

# GUI Tools Guide

This page summarizes the GUI tools distributed with PyBCA/BCL, together with their dependencies, roles, and current implementation status.

Covered tools:

- PyBCA CellSpace Viewer
- BCL Editor
- Rule Editor

## 1. Dependencies

GUI execution requires `PySide6`.
Depending on the workflow, you will also need:

- `numpy`
- `pyyaml`
- `torch`

`pyproject.toml` does not declare GUI-only dependencies, so install them manually.

## 2. PyBCA CellSpace Viewer

Implementation:

- `src/PyBCA/_legacy/guitools.py`
- `src/PyBCA/guitools.py` is only a compatibility wrapper

Start it with:

```bash
python -m PyBCA._legacy.guitools
```

or:

```python
from PyBCA.guitools import main

main()
```

Main capabilities:

- load and save CellSpace YAML
- load, add, clear, and inspect Rule YAML
- load Special Event files
- edit simulation settings
- run one step
- run continuously
- reset

Notes:

- this viewer is still legacy-backed
- `_load_default_files()` may reference paths that do not exist in the current repository layout
- in practice, manual file loading is the safe workflow

## 3. BCL Editor

Implementation:

- `src/BCL/editor.py`

Start it with:

```bash
bcl-editor
```

This command is exposed via `project.gui-scripts` in `pyproject.toml`.

Main capabilities:

- create, open, and save `.bcl`
- place points, rectangles, and lines
- define `element` blocks
- place `element` instances
- copy, cut, and paste selections
- export YAML
- open the Rule Editor

Common shortcuts:

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

Implementation:

- `src/BCL/rule_editor.py`

Start it with:

```bash
python -m BCL.rule_editor
```

Notes:

- it is not currently registered in `project.scripts`
- module execution is the expected entry point

Main capabilities:

- load and save Rule YAML
- edit prev/next patterns
- add and remove rules
- edit direct `probability` values
- preserve `constants`-based alias references

## 5. How GUI Relates To CLI/API

- the BCL Editor exports YAML through `BCLCompiler`
- generated YAML can be passed directly into `Engine` as `cellspace_path`
- Rule YAML created in the Rule Editor can be passed directly into `Config.rule_paths`
- CellSpace assets reviewed in the Viewer can be reused in API-based runs

## 6. Responsibility Split In The Current Implementation

- simulation core: `src/PyBCA/core`
- public execution API: `src/PyBCA/api`
- GUI Viewer: `src/PyBCA/_legacy/guitools.py`
- BCL editing GUI: `src/BCL/editor.py`
- Rule editing GUI: `src/BCL/rule_editor.py`

The GUI surface has not been fully migrated to the new API layout.
In particular, the CellSpace Viewer still depends on the legacy implementation.
