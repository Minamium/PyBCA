---
title: Engine API
parent: English
nav_order: 12
---

# Engine API

`Engine` is the current top-level execution API for PyBCA.
Configuration lives in `Config`, and `Engine.run()` returns a `Result`.

For the actual update order and conflict-resolution logic, see [Simulation Logic](simulation_logic.md).

This page describes the externally visible contract of `Engine` and `Config` as public interfaces.

Implementation references:

- `src/PyBCA/api/config.py`
- `src/PyBCA/api/engine.py`
- `src/PyBCA/api/result.py`
- `src/PyBCA/core/states/state_bca.py`
- `src/PyBCA/core/schemes/bca_default.py`

## 1. Quick Start

```python
from PyBCA.api import Config, Engine

cfg = Config(
    cellspace_path="Sample/Cellspace/C-Join.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    device="cpu",
    trials=3,
    steps=20,
    global_prob=0.5,
    seed=7,
    use_tqdm="false",
)

result = Engine(cfg).run()
print(result.current_step, result.elapsed_sec)
```

## 2. `Config` Field Reference

| Field | Default | Type / Meaning |
| --- | --- | --- |
| `model` | `bca` | `Model` or string |
| `scheme` | `default` | `Scheme` or string |
| `backend` | `torch` | `Backend` or string |
| `cellspace_path` | required | CellSpace YAML path |
| `rule_paths` | required | sequence of Rule YAML paths |
| `device` | `cuda` | torch device string |
| `trials` | `1` | number of parallel trials |
| `steps` | `1` | number of simulation steps |
| `global_prob` | `1.0` | global acceptance probability |
| `seed` | `0` | random seed |
| `spatial_event_file_path` | `None` | Special Event `.py` |
| `gui_mode` | `False` | GUI-oriented return path flag |
| `use_tqdm` | `true` | string enum |
| `trial_constant_sweep` | `None` | alias-based per-trial probability schedule |
| `state_gate_enable` | `False` | enable state gate |
| `state_gate_interval` | `500` | state-gate interval |
| `debug` | `False` | passed to `simulator.step()` |
| `debug_per_trial` | `False` | per-trial debug flag |
| `log_level` | `info` | `debug/info/warning/error` |
| `event_history_path` | `None` | export destination |
| `event_history_format` | `jsonl_trials` | export format |
| `event_history_deduplicate` | `True` | remove duplicate steps |
| `event_history_return_df` | `False` | return DataFrame from export helper |
| `distributed_mode` | `off` | `off/auto/torchrun` |
| `distributed_backend` | `auto` | `auto/nccl/gloo` |
| `distributed_partition` | `block` | trial partition mode; currently only block |
| `distributed_run_dir` | `None` | directory for rank JSON files and shards |
| `distributed_record_configs` | `True` | whether to persist per-rank config JSON |
| `distributed_merge_event_history` | `True` | whether rank shards are merged into a final export |
| `distributed_seed_stride` | `10000019` | per-rank seed offset |

## 3. Accepted Aliases

`Config.__post_init__()` accepts the following aliases.

- `model`: `default -> bca`
- `scheme`: `bca -> default`
- `backend`: `pytorch -> torch`
- `use_tqdm`: `1 -> true`, `0 -> false`
- `log_level`: `warn -> warning`, `err -> error`

`use_tqdm` is parsed as a string enum, not as a Python bool.

## 4. Validation Rules

`Config` validates the following constraints when it is created.

- `cellspace_path` must not be empty
- `rule_paths` must contain at least one file
- `trials >= 1`
- `steps >= 0`
- `global_prob` must be in `[0, 1]`
- `state_gate_interval >= 1`
- `distributed_mode in {off, auto, torchrun}`
- `distributed_backend in {auto, nccl, gloo}`
- `distributed_partition == block`
- `distributed_seed_stride >= 1`

Notes:

- `device` is not validated against current torch availability
- path existence is checked later when files are actually loaded

## 5. `Engine.__init__`

```python
engine = Engine(config, apply_logging_flag=True)
```

Behavior:

1. Optionally call `apply_logging(config)`
2. Build the simulator state with `build_state(config)`
3. Build the step function with `build_stepper(config, state)`

Set `apply_logging_flag=False` if you want to control the logger externally.

## 6. Exact `Engine.run()` Flow

`Engine.run()` executes in this order.

1. Get `simulator` from the built state
2. If `steps > 0`, iterate over `range(steps)`
3. Wrap the iterator with `tqdm` when `use_tqdm == true`
4. Call `self.stepper(step_idx)` on every iteration
5. If `event_history_path` is set, call `save_event_histry_for_dataframe(...)`
6. Return `Result`

Important details:

- `step_idx` is not used by the current default stepper
- the actual step logic is `simulator.step(...)`
- `Result.current_step` is read from `simulator._current_step`

## 7. `Result`

`Result` contains:

- `simulator`
- `current_step`
- `elapsed_sec`
- `event_history`
- `meta`

`meta` currently stores `{"config": config.as_dict}`.

Be precise about `event_history`:

- if `event_history_path is None`, `Result.event_history` is `None`
- raw per-trial histories still remain on `result.simulator.event_history`
- if you want the export helper return value in memory, enable `event_history_return_df=True`

## 8. Special Event Files

The `.py` file referenced by `spatial_event_file_path` must define `events`.

Basic form:

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state),
]
```

Extended form:

```python
events = [
    ("name", (x, y), ref_state, (wx, wy), write_state, prob, start_step, end_step),
]
```

Special events are applied after the regular CA update via `apply_spatial_events()`.

## 9. `trial_constant_sweep`

`trial_constant_sweep` creates per-trial probability vectors for rule aliases defined as `probability: *alias` in Rule YAML files.

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/BNN.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    trials=4,
    steps=15,
    trial_constant_sweep={
        "join_err_0_input": {"base": 0.0, "delta": 0.001},
        "join_err_1_input": {"base": 0.0, "delta": 0.0005},
    },
)
```

Internally, `set_ParallelTrial()` expands this into a `[T, N]` `rule_probs_tensor`.

## 10. `event_history` Export Formats

`event_history_format` accepts:

- `jsonl_trials`
- `jsonl_trials_dict`
- `jsonl`
- `csv`
- `yaml`
- `parquet`

The `Engine` default is `jsonl_trials`.
The underlying simulator helper `save_event_histry_for_dataframe()` still defaults to `parquet`.
If you call the simulator directly, remember that these defaults are different.

## 11. Trial Distribution with `torchrun`

When `distributed_mode="torchrun"`, each rank launches its own `Engine` instance and processes a block-partitioned subset of the global trials.

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/Join_err/P0_join.yaml",
    rule_paths=("Sample/rule/base-rule.yaml", "Sample/rule/Join_fork.yaml"),
    spatial_event_file_path="Sample/Specialevent/Join_detect.py",
    device="cuda",
    trials=2000,
    steps=500,
    event_history_path="out/join.jsonl",
    event_history_format="jsonl_trials",
    distributed_mode="torchrun",
    distributed_run_dir="out/join.dist",
)
```

Example launcher:

```bash
python -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29541 \
  --nproc_per_node=8 \
  your_script.py
```

The public behavior in distributed mode is:

- each rank receives a resolved local `Config`
- `device="cuda"` is rewritten as `cuda:{LOCAL_RANK}`
- `trial_constant_sweep` is shifted by the rank-local `trial_offset`
- when `event_history_path` is set, each rank writes a shard and rank 0 merges the final output
- when `distributed_record_configs=True`, each resolved rank config is written to `distributed_run_dir/rank_configs/rank_XXXX.json`

`Result.meta["distributed"]` stores rank information, trial ranges, manifest paths, shard paths, and the merged output path.
Even when `event_history_return_df=True`, the returned DataFrame remains rank-local; the merged output path is exposed via `Result.meta["distributed"]["paths"]["event_history_merged_path"]`.

## 12. Logging

`apply_logging(config)` configures the `PyBCA` logger with a single `StreamHandler` and the requested `log_level`.

Default formatter:

```text
%(levelname)s : %(message)s
```

## 13. Common Usage Patterns

### Make paths absolute

```python
from pathlib import Path
from PyBCA.api import Config

root = Path(__file__).resolve().parents[1]
cfg = Config(
    cellspace_path=str(root / "Sample" / "Cellspace" / "C-Join.yaml"),
    rule_paths=(str(root / "Sample" / "rule" / "base-rule.yaml"),),
)
```

### Get an in-memory event-history DataFrame

```python
result = Engine(cfg).run()
df = result.simulator.save_event_histry_for_dataframe(path=None, return_df=True)
```

### Disable the progress bar

```python
cfg = Config(
    cellspace_path="Sample/Cellspace/test.yaml",
    rule_paths=("Sample/rule/base-rule.yaml",),
    use_tqdm="false",
)
```
