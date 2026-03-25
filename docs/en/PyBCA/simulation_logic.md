---
title: Simulation Logic
parent: English
nav_order: 13
---

# Simulation Logic

This page describes the one-step update mechanism implemented in `src/PyBCA/core/simulator.py`.
The emphasis is not on the public API surface, but on state representation, rule matching, probabilistic gating, conflict resolution, state conversion, and special-event application.

## 1. State Representation And Input Assets

PyBCA represents a CellSpace as a two-dimensional lattice of integer-valued states.
The basic states are:

- `0`: vacant
- `1`: wire
- `2`: token
- `-1`: recycle-bin connection point

The primary input assets are:

- CellSpace YAML
- Rule YAML
- Special Event `.py`

CellSpace YAML is cropped to its minimal bounding rectangle and stored together with offset information.
Accordingly, the coordinate system in the source file is not always identical to the internal tensor index system.

## 2. Internal Tensor Representation

When `set_ParallelTrial(T)` is called, a single CellSpace is replicated into `T` trials.
The main tensors can be understood as follows:

- `TCHW`: `[Trial, 1, Height, Width]`
- `rule_arrays_tensor`: `[N, 2, 3, 3]`
- `rule_probs_tensor`: `[N]` or `[T, N]`
- `TNHW_boolMask`: `[Trial, N, Height, Width]`
- `TCHW_applied`: `[Trial, 1, Height, Width]`

Here `N` denotes the number of loaded transition rules.
When `trial_constant_sweep` is active, `rule_probs_tensor` is expanded to `[T, N]` so that different trials may carry different rule probabilities.

## 3. Computation Order Of One Step

1. initialize the random generator as `self._current_step + 65536 + seed`
2. reset `TNHW_boolMask`, `tmp_mask`, and `TCHW_applied`
3. compute candidate rule centers through `_match_centers_all_rules()`
4. apply the global probability gate through `_global_prob_gate()`
5. apply per-rule probability gating through `_rule_prob_gate()`
6. generate one shuffled rule order with `torch.randperm(N)`
7. for each rule in that order, perform inner-conflict resolution, outer-conflict resolution, and write-back
8. optionally apply `apply_state_gates()`
9. optionally apply `apply_spatial_events()`
10. increment `current_step`

An important implementation detail is that the shuffled rule order is shared across all trials within the same step.

## 4. Rule Matching

Each rule is stored as a 3x3 pair of `prev` and `next` patterns.
Matching is performed in `_match_centers_all_rules()`.

The implementation has the following consequences:

- out-of-bounds regions are zero-padded before matching
- the cross mask `[[0,1,0],[1,1,1],[0,1,0]]` is always active
- corner cells are matched only when the corresponding `prev` corner entry is nonzero
- active positions are compared by strict equality
- `-1` is not a wildcard; when it is part of the active mask, it is matched as an ordinary state value

Therefore, a corner value of `0` in `prev` does not necessarily impose a strict zero condition.
Under the present implementation, zero-valued corners behave effectively as unspecified positions.

## 5. Global Probability Gate

`global_prob` is applied before the rule-specific probabilities.
`_global_prob_gate()` accepts the following shapes:

- scalar
- `[N]`
- `[T, N]`
- `[T, N, H, W]`
- `None`

`None` means that the gate is skipped.
Otherwise, the value is normalized to `[T, N, H, W]`, and an independent Bernoulli decision is drawn for each candidate site.

## 6. Rule-Specific Probability And Sweep Semantics

Rule probabilities are loaded from the `probability` field in Rule YAML.
When `trial_constant_sweep` is provided, all rules associated with a YAML alias `probability: *alias` receive the trial-wise schedule

`base + trial_index * delta`

after clamping to `[0, 1]`.

This mechanism represents probability parameters, not temperature itself.
Any effective-temperature interpretation is introduced at the analysis level rather than in the simulator core.

## 7. Conflict Resolution

PyBCA removes update conflicts in two stages.

### 7.1 Intra-rule conflicts

`_rule_inner_conflict_resolution()` removes center candidates when multiple candidates of the same rule would write to the same target cell.

### 7.2 Inter-rule conflicts

`_rule_outer_conflict_resolution()` removes candidates that overlap with cells already marked in `TCHW_applied`.

After these two stages, `_write_back()` is executed under the assumption that the remaining target writes are non-conflicting.

## 8. Write-Back Semantics

Only differential cells are written back.
Concretely, the write mask is defined by

- `post != pre`
- `post != -1`

Hence, `-1` in the `next` pattern does not mean “write the state `-1`”.
Instead, it functions as a sentinel meaning “do not write this position back”.

## 9. State Gate

`apply_state_gates()` reads the `conv` sections from Rule YAML, constructs a lookup table over the `int8` state domain, and applies all conversions simultaneously.

Important consequences are:

- conversion is simultaneous rather than sequential
- when the same `prev` state is defined multiple times, the last definition wins
- the gate is applied when `state_gate_enable=True` and `_current_step % state_gate_interval == 0`
- because the check occurs before incrementing `current_step`, the gate may also apply at step `0`

## 10. Spatial Events

`apply_spatial_events()` is executed after the ordinary local-rule update.
Each event row may follow either the legacy 6-column format or the extended 9-column format.

In the extended format, the following can be specified:

- event probability
- `start_step`
- `end_step`

Implementation details:

- coordinates are defined in the external global coordinate system
- at runtime they are translated into local tensor indices by subtracting `offset_x` and `offset_y`
- out-of-range events are discarded
- `start_step` and `end_step` are interpreted inclusively
- the event probability gate is evaluated independently per trial
- event writes also update `TCHW_applied`

## 11. Event History And Metadata

If special-event names are present, `set_ParallelTrial()` initializes `event_history` as

- `List[Dict[str, List[int]]]`

`save_event_histry_for_dataframe()` can then export the history together with metadata such as:

- `parallel_trial`
- `current_step`
- `device`
- `offset_x`, `offset_y`
- `rule_ids`
- `rule_probs_base`
- `probability_sweep`

This enables downstream analyses that jointly use event timing and per-trial sweep conditions.

## 12. Interpreting The BNN Experiments

In `tests/BNN.py`, the variables `join_err_*` and `fork_err_*` are swept across trials while events such as

- `output`
- `add_weight`
- `subs_weight_pre`
- `subs_weight_post`

are observed.

What PyBCA directly provides is a probability sweep together with an event history.
Whether those data are interpreted as performance curves against an effective temperature depends on an external theoretical mapping.

## 13. Performance-Relevant Parameters

The parameters with the strongest practical effect on runtime are:

- `device`
- `trials`
- `steps`
- `use_tqdm`

For large grids or many trials, `device="cuda"` is typically decisive.
For rare-event studies, sufficiently long `steps` are essential.

## 14. Where To Go Next

- usage and workflow: [PyBCA Guide](PyBCA_guide.md)
- argument reference: [Engine API](engine_api.md)
- BCL syntax: [BCL Guide](../bcl/0.1/guide.md)
