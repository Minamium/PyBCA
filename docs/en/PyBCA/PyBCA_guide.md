---
title: PyBCA Guide
parent: English
nav_order: 11
---

# PyBCA Guide

This guide is the main entry point for running PyBCA with the current implementation.
For new code, use `PyBCA.api.Engine`. Treat the legacy API as a compatibility surface.

Related pages:

- [Engine API](engine_api.md)
- [GUI Tools Guide](guitools_guide.md)
- [Implementation Parity Audit](parity_audit.md)
- [BCL Guide](../bcl/0.1/guide.md)

## 1. Package Layout

- `src/PyBCA/api`
  Public API layer with `Config`, `Engine`, and `Result`.
- `src/PyBCA/core`
  Actual simulator implementation plus state/scheme builders.
- `src/PyBCA/_legacy`
  Shelved pre-refactor implementation kept for compatibility and validation.
- `src/BCL`
  BCL compiler, editor, and rule editor.

## 2. Installation And Dependencies

Base dependencies are defined in `pyproject.toml`.

```bash
pip install -e .
```

Main runtime dependencies:

- `numpy`
- `torch>=2.0`
- `scipy`
- `pyyaml`

If you want the GUI tools, install `PySide6` separately.

```bash
pip install PySide6
```

If you run directly from the repository without installation, add `PYTHONPATH=src`.

## 3. Recommended Workflow

1. Prepare a CellSpace YAML file
2. Provide one or more Rule YAML files
3. Optionally provide a Special Event `.py`
4. Build a `Config`
5. Run `Engine.run()`
6. Optionally export event history with `event_history_path`

If you use BCL, compile `.bcl` into YAML first and then pass the generated YAML to `Config.cellspace_path`.

## 4. Minimal Example

```python
from pathlib import Path

from PyBCA.api import Config, Engine

root = Path.cwd()

cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
    device="cpu",
    trials=3,
    steps=20,
    global_prob=0.5,
    seed=7,
    use_tqdm="false",
)

result = Engine(cfg).run()
print(result.current_step)
```

Notes:

- `Config` does not normalize paths
- the default `device` is `"cuda"`
- `steps=0` initializes the simulator without advancing steps

## 5. `run(dict)` Shortcut

```python
from PyBCA.api import run

result = run(
    {
        "cellspace_path": "Sample/Cellspace/test.yaml",
        "rule_paths": ["Sample/rule/base-rule.yaml"],
        "device": "cpu",
        "trials": 2,
        "steps": 8,
        "global_prob": 0.5,
        "seed": 13,
        "use_tqdm": "false",
    }
)
```

Dictionary input is still converted through `Config(**dict)`.

## 6. Special Events And Event History

Set `spatial_event_file_path` to enable special events.
Set `event_history_path` and `event_history_format` to export event histories.

```python
from PyBCA.api import Config, Engine

cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    spatial_event_file_path="Sample/Specialevent/test_event.py",
    trials=2,
    steps=110,
    event_history_path="/tmp/test_event.jsonl",
    event_history_format="jsonl_trials",
    event_history_deduplicate=True,
    event_history_return_df=True,
    use_tqdm="false",
)

result = Engine(cfg).run()
print(type(result.event_history))
```

Strict behavior:

- `simulator.event_history` is initialized during `set_ParallelTrial()`
- `Engine.run()` only calls `save_event_histry_for_dataframe(...)` if `event_history_path` is set
- therefore `Result.event_history` reflects the export helper return value, not the raw internal history
- raw per-trial histories remain available on `result.simulator.event_history`

## 7. Trial-Wise Probability Sweep

If your Rule YAML uses `probability: *alias`, you can create trial-wise probability schedules with `trial_constant_sweep`.

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/BNN.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    trials=4,
    steps=15,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.001},
        "join_err_1_input": {"base": 0.0, "delta": 0.0005},
        "fork_err_0_input": {"base": 0.0, "delta": 0.001},
    },
)
```

Alias names must match the loaded Rule YAML constants. Otherwise PyBCA raises `ValueError`.

## 8. Legacy Compatibility API

`src/PyBCA/cli_simClass.py` exposes both the current simulator and the legacy simulator.

```python
from PyBCA.cli_simClass import BCA_Simulator, LegacyBCA_Simulator

sim = BCA_Simulator(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=["Sample/rule/base-rule.yaml"],
    device="cpu",
)
sim.Allocate_torch_Tensors_on_Device()
sim.set_ParallelTrial(3)
sim.run_steps(steps=20, global_prob=0.5)
```

Interpretation:

- `BCA_Simulator`: `src/PyBCA/core/simulator.py`
- `LegacyBCA_Simulator`: `src/PyBCA/_legacy/cli_simClass.py`
- `PyBCA.lib`: still re-exports legacy utility implementations
- `PyBCA.guitools`: still re-exports legacy GUI implementations

## 9. Sample Asset Locations

- CellSpace YAML: `Sample/Cellspace/*.yaml`
- Rule YAML: `Sample/rule/*.yaml`
- Special Events: `Sample/Specialevent/*.py`
- BCL sources: `Sample/bclfile/*.bcl`

Useful validation scripts:

- parity: `PYTHONPATH=src python tests/simulator_parity.py`
- sample runners:
  `tests/BNN.py`,
  `tests/BCA-IP.py`,
  `tests/Join_acc.py`,
  `tests/Fork_acc.py`

## 10. Where To Look Next

- Configuration details: [Engine API](engine_api.md)
- BCL syntax: [BCL Guide](../bcl/0.1/guide.md)
- GUI operations: [GUI Tools Guide](guitools_guide.md)
- legacy/new consistency status: [Implementation Parity Audit](parity_audit.md)
