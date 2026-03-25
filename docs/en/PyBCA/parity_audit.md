---
title: Implementation Parity Audit
nav_exclude: true
---

# Implementation Parity Audit

This page records the consistency audit between the legacy implementation and the refactored implementation as of 2026-03-26.

## 1. Audit Scope

- `src/PyBCA/_legacy/cli_simClass.py`
- `src/PyBCA/core/simulator.py`
- `src/PyBCA/_legacy/lib.py`
- `src/PyBCA/core/io.py`
- `src/PyBCA/api/*`
- `src/PyBCA/cli_simClass.py`
- `src/PyBCA/lib.py`
- `src/PyBCA/guitools.py`
- `src/BCL/compiler.py`
- `tests/simulator_parity.py`

## 2. Direct Comparison Conclusions

### `core/simulator.py` vs the legacy simulator

- `src/PyBCA/core/simulator.py` preserves the legacy update logic while cleaning up imports and documentation
- `BCASimulator = BCA_Simulator` was added as an alias
- parity was checked for single-step execution, `run_steps`, and `Engine.run()`

### `core/io.py` vs legacy `lib.py`

- they are effectively identical at audit time
- no differences were observed on the tested CellSpace/Rule/Event loading surface

### Public wrappers

- `src/PyBCA/cli_simClass.py` exposes the new core simulator and also re-exports `LegacyBCA_Simulator`
- `src/PyBCA/lib.py` still wraps the legacy utility implementation
- `src/PyBCA/guitools.py` still wraps the legacy GUI implementation

This is intentional for compatibility. It is not a bug, but it does mean the migration is not complete across every public surface.

## 3. Runtime Validation

The following CPU validation was executed successfully.

```bash
PYTHONPATH=src python tests/simulator_parity.py
```

Coverage:

- 20 cases
- step-by-step comparison
- `run_steps()` comparison
- `Engine.run()` comparison
- `event_history` comparison
- export comparison for `jsonl_trials`, `jsonl_trials_dict`, and `jsonl`
- samples covering `BNN`, `Join/Fork`, `C-Join err`, `BCA-IP`, and special events

Result:

- the full current parity suite passed

## 4. BCL-Side Consistency

The following checks were also confirmed.

- every `Sample/bclfile/*.bcl` file parsed successfully
- `Sample/bclfile/BNN.bcl` matched `Sample/Cellspace/BNN.yaml`
- `Sample/bclfile/C-Join_from_JF.bcl` matched `Sample/Cellspace/2-way_C-Join.yaml`

So within the current sample surface, the BCL compiler is aligned with the checked CellSpace YAML assets.

## 5. Current Boundaries

- parity is confirmed on CPU
- GPU parity was not re-run within this audit page scope
- the GUI Viewer still depends on the legacy implementation
- the utility helper surface still re-exports legacy code

## 6. Practical Interpretation

For new execution code, `PyBCA.api.Engine` is safe to use.
At the same time, the accurate description of the repository state is:

- CA update theory and sample execution results are in parity
- some public APIs are still compatibility wrappers
- GUI migration is not complete
