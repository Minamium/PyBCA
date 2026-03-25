---
title: PyBCA Dev Memo
nav_exclude: true
---

# PyBCA Dev Memo

This memo captures the intended simulation theory and update algorithm behind PyBCA.

## 1. Target Model

PyBCA approximates a 2D Brownian circuit as a cellular automaton.

Main states:

- `0`: vacant
- `1`: wire
- `2`: token
- `-1`: recycle-bin connection point

A recycle-bin connection point is treated as an idealized token source or sink.

## 2. Central Implementation Objects

- `BCA_Simulator`
  Loads CellSpace and Rule YAML, then runs updates on `torch`.
- CellSpace Viewer
  Used for visualization and manual inspection of CellSpace assets.

## 3. Goal Of A Single Step Update

The step update is designed to keep trial-level and cell-level GPU parallelism while preserving:

- fair randomness
- conflict-safe updates
- prevention of invalid token duplication

## 4. Main Tensors

- `TCHW`
  CellSpace tensor with shape `[Trial, 1, H, W]`
- `rule_arrays`
  Transition rules with shape `[N, 2, 3, 3]`
- `rule_mask`
  Neighborhood mask used for rule matching
- `rule_probs`
  Rule-wise probabilities
- `TNHW_boolMask`
  Mask of centers where each rule can apply
- `tmp_mask`
  Scratch tensor for overlap detection
- `TCHW_applied`
  Flag tensor marking cells already updated in the current step

## 5. Update Algorithm

1. initialize the random generator from the current step and seed
2. compute rule matches into `TNHW_boolMask`
3. apply the global probability gate
4. process rules in shuffled order
5. resolve intra-rule conflicts
6. resolve conflicts against already-applied updates
7. write the `next` pattern into `TCHW`
8. optionally apply the state gate
9. optionally apply spatial events

## 6. Why Conflict Resolution Exists

If multiple rules write to the same cell within one step, the CA interpretation becomes order-sensitive.
PyBCA projects candidate writes into `tmp_mask` and drops overlapping candidates to keep the update consistent under parallel execution.

## 7. State Gates And Spatial Events

- state gate
  A periodic global state conversion mechanism.
- spatial event
  A conditional write mechanism based on cell states at specific positions.

These are separate from the normal local rule update, but they are important for experiment-driven models.

## 8. Current Migration Status

- the simulation core has been migrated to `src/PyBCA/core/simulator.py`
- the GUI Viewer still uses the legacy implementation
- utility helpers still expose legacy-backed wrappers

The execution theory itself is parity-checked against the legacy simulator on the current test surface.
