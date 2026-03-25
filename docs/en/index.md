---
title: English
nav_order: 20
has_children: true
permalink: /en/
---

# English Documentation

Last updated: 2026-03-26

PyBCA is an open-source Brownian Cellular Automaton simulator with an accompanying BCL cellspace description language.
For new code, the recommended execution entry point is `PyBCA.api.Engine`.

## Main Pages

- [PyBCA Guide](PyBCA/PyBCA_guide.md)
- [Simulation Logic](PyBCA/simulation_logic.md)
- [Engine API](PyBCA/engine_api.md)
- [GUI Tools Guide](PyBCA/guitools_guide.md)
- [BCL Guide](bcl/0.1/guide.md)

## Documentation Policy

- put usage first
- explain the update and computation logic explicitly
- treat BCL and GUI as tooling around the simulation workflow
- keep migration and parity notes as supplementary material

## Common Entry Points

- New execution code: `PyBCA.api.Engine`
- Legacy-compatible simulator import: `PyBCA.cli_simClass.BCA_Simulator`
- BCL compilation: `bcl INPUT.bcl -o OUTPUT.yaml`
- GUI tools: `bcl-editor`, `python -m BCL.rule_editor`, `python -m PyBCA._legacy.guitools`

## Supplementary Notes

- [Implementation Parity Audit](PyBCA/parity_audit.md)
- [PyBCA Dev Memo](dev-memo/PyBCAdevMemo.md)
