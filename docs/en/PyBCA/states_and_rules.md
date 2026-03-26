---
title: States and Rules
parent: English
nav_order: 14
---

# States and Rule Families

This page explains how PyBCA treats state values, how additional states can be used to build custom transition systems, how to read the bundled `Join/Fork` rule set, and how far the current GUI tools can be used for design and simulation setup.

Related pages

- [PyBCA Guide](PyBCA_guide.md)
- [Simulation Logic](simulation_logic.md)
- [GUI Tools Guide](guitools_guide.md)

## 1. The Implemented State Model

In the implementation, both CellSpace data and Rule patterns are integer-valued grids.
Accordingly, `Join`, `Fork`, and similar devices are not built-in primitives in the simulation kernel. They are expressed as families of local rules over additional integer states.

The important implementation facts are:

- `0` is the standard vacant state
- `-1` may appear in CellSpace and `prev` patterns as an ordinary matched integer state
- however, `-1` on the `next` side does not mean “write state `-1`”; it is treated as a write-back exclusion sentinel
- the internal tensor dtype is `int8`, so the practical state range is `-128..127`

Therefore, aside from the special handling of `0` and `-1`, additional states are user-defined integers whose meaning is determined entirely by the model author.

## 2. Basic States and Sample Conventions

The primary public-state convention is:

| State | Usual meaning |
| --- | --- |
| `0` | vacant |
| `1` | wire |
| `2` | token |
| `-1` | recycle-bin connection point |

The bundled samples then introduce additional internal states.
In `Sample/rule/Join_fork.yaml`, the working convention is:

| State | Conventional role in `Join_fork.yaml` |
| --- | --- |
| `3` | stable center state for Join-like devices |
| `4` | transient post-transition Join state |
| `5` | stable center state for Fork-like devices |
| `6` | transient post-transition Fork state |

These meanings are sample-level conventions, not kernel-level reservations.
You are free to assign different meanings to different integers in other models.

## 3. Rule YAML Usually Represents a Rule Family

A Brownian-circuit device is often described not by a single local rule but by a family of rules covering orientation, error modes, and transient-state relaxation.

The main sections in Rule YAML are:

- `constants`
  named probability parameters
- `conv`
  global state conversions applied through the state gate
- `rules`
  the local `prev -> next` transition rules

This separation lets a model treat the following layers independently:

- CellSpace layout and initial states
- local neighborhood transitions
- relaxation from transient states back to stable internal states

## 4. How To Read `Join_fork.yaml`

`Sample/rule/Join_fork.yaml` is the main bundled example of an extended-state rule family.

### 4.1 Probability Constants

The `constants` section defines the following aliases:

- `join_err_0_input`
- `join_err_1_input`
- `fork_err_0_input`

They are referenced inside the rule list as `probability: *alias`, and can later be swept via `Config.trial_constant_sweep`.

### 4.2 Transient-State Relaxation

The `conv` section defines:

- `4 -> 3`
- `6 -> 5`

This means that the sample uses explicit transient states immediately after Join/Fork transitions, and then relaxes them back to stable center states through the state gate.
As a result, `Join_fork.yaml` is normally used together with `state_gate_enable=True`.

### 4.3 Rule-Family Structure

The bundled rules are grouped by direction:

- `200-203`: right-facing Join
- `204-205`: right-facing Fork
- `206-209`: upward Join
- `210-211`: upward Fork
- `212-215`: left-facing Join
- `216-217`: left-facing Fork
- `218-221`: downward Join
- `222-223`: downward Fork

So a `Join` or `Fork` device is not triggered simply because the center cell happens to contain `3` or `5`.
Its behavior is defined by a family of rules that jointly encode center states, input/output orientation, token placements, error configurations, and probability parameters.

### 4.4 Error Rules

The rules that reference `join_err_*` and `fork_err_*` are probabilistic error variants around the nominal transition rules.
They encode the idea that perturbed local configurations may still be accepted with a tunable probability.

PyBCA itself only provides probability-controlled local transitions.
Any interpretation of these probabilities as temperature or thermal noise belongs to the external analysis model, not to the simulator core.

## 5. Designing Arbitrary Additional States

To define a custom device with extra states, the minimal design procedure is:

1. choose unused integer states
2. place those states in the CellSpace
3. define the corresponding `prev` and `next` patterns in Rule YAML
4. if transient states must relax back to stable states, define `conv` and enable the state gate
5. if probability sweeps are needed, define aliases in `constants`

Important constraints:

- left/right and up/down symmetries are not generated automatically; mirrored rules must be written explicitly
- `0` is best kept for vacant cells because it also interacts with omission-style rule notation
- `next = -1` should not be used as an ordinary target state because it is interpreted as “do not write back”

## 6. Customizability and Limits of the Editors

### 6.1 BCL Editor

The BCL Editor is primarily a CellSpace and element-layout editor.

It supports:

- point, rectangle, and line placement
- `element` definitions and `place.Element(...)` placement
- custom cell-state values
- dynamic canvas expansion

Implemented constraints:

- the preset brush list exposes `0, 1, 2, -1, 3, 4`
- other values can still be placed through the `Custom` field in the range `-128..127`
- the BCL source pane is read-only, so this is not a full free-form text editor

In practice, the BCL Editor is strong for CellSpace construction, but it is not the main tool for authoring complex rule families.

### 6.2 Rule Editor

The Rule Editor is focused on local-rule editing.

It supports:

- editing `prev` and `next` patterns
- adding and deleting rules
- changing pattern size
- using arbitrary integer states
- direct editing of `probability`
- preserving and selecting existing `constants` aliases

Implemented constraints:

- pattern size is limited to `3x3`, `5x5`, `7x7`, `9x9`
- the custom brush range is `-128..127`
- `constants` and `conv` are preserved on load/save
- but there is no dedicated GUI for systematically authoring new `constants` or `conv` sections from scratch

So the Rule Editor is flexible for local-pattern design, but full rule-file authorship still benefits from hand-editing YAML when the model structure becomes more elaborate.

### 6.3 CellSpace Viewer

The CellSpace Viewer is intended for interactive inspection of one simulation setup at a time.

It supports:

- loading CellSpace, Rule, and Special Event files
- changing `global_prob`, `seed`, `device`, and state-gate settings
- single-step and continuous execution

Current limits:

- it still depends on the legacy backend
- GUI mode fixes the number of trials to `1`
- it is not the main interface for `trial_constant_sweep` or distributed execution

Accordingly, the Viewer is useful for qualitative inspection, but not for large sweeps, HPC jobs, or distributed trial orchestration.

## 7. Which Tool To Use

- design CellSpace layout and device placement: `BCL Editor`
- edit local rule families: `Rule Editor`
- inspect one simulation interactively: `CellSpace Viewer`
- run large trial counts, parameter sweeps, or distributed jobs: `Engine` and scripts

For research use, the most robust workflow is to treat the GUI as a front-end for design and inspection, and move the final experiment execution to script-based runs through `PyBCA.api.Engine`.
