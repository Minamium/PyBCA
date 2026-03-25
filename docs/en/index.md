---
title: English
nav_order: 20
has_children: true
permalink: /en/
---

# English Documentation

Audit baseline date: 2026-03-26

PyBCA is a Brownian Cellular Automaton simulator with an accompanying BCL cellspace description language.
The recommended execution path is `PyBCA.api.Engine`.

## Main Pages

- [PyBCA Guide](PyBCA/PyBCA_guide.md)
- [Engine API](PyBCA/engine_api.md)
- [GUI Tools Guide](PyBCA/guitools_guide.md)
- [Implementation Parity Audit](PyBCA/parity_audit.md)
- [BCL Guide](bcl/0.1/guide.md)
- [PyBCA Dev Memo](dev-memo/PyBCAdevMemo.md)

## Audit Highlights

- `src/PyBCA/core/simulator.py` is the migrated simulator that preserves the legacy cellular automaton update semantics.
- `src/PyBCA/api` is a thin orchestration layer over `core`.
- `src/PyBCA/lib.py` and `src/PyBCA/guitools.py` are still legacy-backed compatibility wrappers.
- CPU parity has been validated with `tests/simulator_parity.py`.

## Common Entry Points

- New execution code: `PyBCA.api.Engine`
- Legacy-compatible simulator import: `PyBCA.cli_simClass.BCA_Simulator`
- BCL compilation: `bcl INPUT.bcl -o OUTPUT.yaml`
- GUI tools: `bcl-editor`, `python -m BCL.rule_editor`, `python -m PyBCA._legacy.guitools`
