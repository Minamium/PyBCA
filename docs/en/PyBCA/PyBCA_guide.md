---
title: PyBCA Guide
parent: English
nav_order: 11
---

# PyBCA Guide

This guide serves as an introductory document for configuring and running numerical experiments with PyBCA.
It begins with the standard execution workflow and then points to the computation logic and API details.

Related pages:

- [Simulation Logic](simulation_logic.md)
- [States and Rules](states_and_rules.md)
- [Engine API](engine_api.md)
- [GUI Tools Guide](guitools_guide.md)
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

## 3. Standard Execution Procedure

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

## 6. Inputs And Outputs

- CellSpace input:
  `Config.cellspace_path`
- Rule input:
  `Config.rule_paths`
- Special Event input:
  `Config.spatial_event_file_path`
- event history output:
  `Config.event_history_path`, `Config.event_history_format`

Common output formats:

- `jsonl_trials`
- `jsonl_trials_dict`
- `jsonl`
- `csv`
- `yaml`
- `parquet`

## 7. Special Events And Event History

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

## 8. Trial-Wise Sweep

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

This mechanism is used for error-rate or acceptance-rate studies.
The simulator itself does not define temperature as an internal variable. Any effective-temperature interpretation belongs to the analysis layer.

For extended-state rule families such as `Join/Fork`, and for the general design of arbitrary additional states, see [States and Rules](states_and_rules.md).

## 9. Computation Logic

The update order, conflict resolution, state gate, and spatial event application order are summarized in [Simulation Logic](simulation_logic.md).

## 10. Sample Asset Locations

- CellSpace YAML: `Sample/Cellspace/*.yaml`
- Rule YAML: `Sample/rule/*.yaml`
- Special Events: `Sample/Specialevent/*.py`
- BCL sources: `Sample/bclfile/*.bcl`

Bundled runner scripts:

- sample runners:
  `tests/BNN.py`,
  `tests/BCA-IP.py`,
  `tests/Join_acc.py`,
  `tests/Fork_acc.py`

## 11. Compatibility API

For compatibility with existing code, `PyBCA.cli_simClass.BCA_Simulator` remains available.
For new code, `PyBCA.api.Engine` is the clearer reference path.

## 12. Where To Look Next

- start from usage: [PyBCA Guide](PyBCA_guide.md)
- understand the update order: [Simulation Logic](simulation_logic.md)
- understand extended states and `Join/Fork` rules: [States and Rules](states_and_rules.md)
- inspect configuration details: [Engine API](engine_api.md)
- learn BCL syntax: [BCL Guide](../bcl/0.1/guide.md)
- use the GUI tools: [GUI Tools Guide](guitools_guide.md)
