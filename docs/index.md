title: PyBCA Docs
nav_order: 1
---

# PyBCA Documentation

This GitHub Pages site now has canonical Japanese and English sections.

Implementation audit summary:

- On March 26, 2026, the migrated simulator and `Engine` path were checked against the legacy simulator on CPU with `PYTHONPATH=src python tests/simulator_parity.py`.
- The current parity suite passed all 20 cases, including stepwise execution, `run_steps`, `Engine.run()`, special-event histories, and event export formats.
- `src/PyBCA/core/simulator.py` and `src/PyBCA/core/io.py` remain behaviorally aligned with the legacy simulator surface used by the tests.
- `src/PyBCA/lib.py` and `src/PyBCA/guitools.py` are still compatibility wrappers backed by legacy modules by design.

## Choose A Language

- Japanese: [日本語ドキュメント](ja/)
- English: [English documentation](en/)

## Documentation Map

- Japanese:
  [Guide](ja/PyBCA/PyBCA_guide.md),
  [Engine API](ja/PyBCA/engine_api.md),
  [GUI Tools](ja/PyBCA/guitools_guide.md),
  [BCL Guide](ja/bcl/0.1/guide.md),
  [Parity Audit](ja/PyBCA/parity_audit.md),
  [Dev Memo](ja/dev-memo/PyBCAdevMemo.md)
- English:
  [Guide](en/PyBCA/PyBCA_guide.md),
  [Engine API](en/PyBCA/engine_api.md),
  [GUI Tools](en/PyBCA/guitools_guide.md),
  [BCL Guide](en/bcl/0.1/guide.md),
  [Parity Audit](en/PyBCA/parity_audit.md),
  [Dev Memo](en/dev-memo/PyBCAdevMemo.md)

## Compatibility Paths

The old top-level pages under `docs/PyBCA/`, `docs/bcl/`, and `docs/dev-memo/` are kept as compatibility entry pages so existing links do not break.
