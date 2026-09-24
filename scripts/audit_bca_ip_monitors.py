"""Audit BCA-IP event predicates and reproduce monitor CJoin phase blocking.

The pulse experiments use cropped copies of the actual cell space. They are
diagnostic interventions, not additional BCA-IP statistical trials. Production
rules, events, cell spaces, and checkpoints are never changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from PyBCA.core.simulator import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]
CELLSPACE = ROOT / "Sample/Cellspace/BCA-IP.yaml"
RULES = ROOT / "Sample/rule/base-rule.yaml"
EVENTS = ROOT / "Sample/Specialevent/BCA-IP_event.py"
# name, inclusive crop, pulse input, downstream sink, real counter, CJoin sites
FIXTURES = [
    ("fsm_x1", (169, 189, 4, 22), (172, 8), (181, 20), (187, 9), [(180, 8), (181, 9)]),
    ("fsm_x2", (169, 189, 70, 88), (172, 74), (181, 86), (187, 75), [(180, 74), (181, 75)]),
    ("input", (-104, -82, 4, 19), (-101, 12), (-92, 7), (-87, 11), [(-93, 12), (-92, 11)]),
    ("amp", (124, 157, 32, 49), (152, 46), (125, 46), (137, 35), [(138, 42), (137, 41)]),
]
SETTINGS = [("global_1", 1., 1.), ("global_1_bins_0.5", 1., .5), ("global_0.5", .5, 1.)]


def make_base(mode):
    return BCA_Simulator(
        str(CELLSPACE), [str(RULES)], device="cpu", quiet=True,
        spatial_event_filePath=str(EVENTS), execution_mode=mode,
        rng_mode="legacy" if mode == "reference" else "independent",
    )


def crop_cells(base, bounds, dy=0):
    x0, x1, y0, y1 = bounds
    return base.cellspace[y0 + dy - base.offset_y:y1 + dy - base.offset_y + 1,
                          x0 - base.offset_x:x1 - base.offset_x + 1].copy()


def check_geometry(base):
    events = dict(zip(base.spatial_event_names, base.spatial_event_arrays))
    layouts = {fixture[0]: fixture for fixture in FIXTURES}
    checked = []
    for unit, unit_dy in [("A", 0), ("B", 495)]:
        for i in range(6):
            for category in ["input", "fsm", "amp"]:
                layout = ("fsm_x1" if i == 0 else "fsm_x2") if category == "fsm" else category
                _, bounds, _, _, counter, _ = layouts[layout]
                dy = unit_dy + 66 * i - (66 if layout == "fsm_x2" else 0)
                name = {"input": f"{unit}_core_input_{i+1}", "fsm": f"{unit}_x{i+1}output",
                        "amp": f"{unit}_Amp_x{i+1}output"}[category]
                same = np.array_equal(crop_cells(base, bounds), crop_cells(base, bounds, dy))
                coordinate = tuple(map(int, events[name][:2]))
                correct = coordinate == (counter[0], counter[1] + dy)
                assert same and correct, (name, "monitor layout or event coordinate changed")
                checked.append({"event": name, "layout": layout, "coordinate": coordinate,
                                "same_initial_patch": same})
    return checked


def check_predicates(base):
    s = copy.copy(base)
    s.trial_ids = [0]
    s.Allocate_torch_Tensors_on_Device()
    s.set_ParallelTrial(1)
    if s.candidate_plan is not None:
        s.candidate_plan.set_seed(20260925)
    rows, names = s.spatial_event_arrays, s.spatial_event_names
    # Six-column event definitions implicitly use probability 1 with no window.
    if rows.shape[1] >= 7:
        assert np.all(rows[:, 6] == 1)
    if rows.shape[1] >= 9:
        assert np.all(rows[:, 7:9] == -1)
    refs = sorted({tuple(map(int, row[:2])) for row in rows})
    cases = [[]] + [[ref] for ref in refs] + [refs]
    for active in cases:
        s.TCHW.copy_(s.cellspace_tensor[None, None])
        s.TCHW_applied.zero_()
        s._current_step = 123
        s.event_history = [{name: [] for name in names}]
        for x, y in active:
            s.TCHW[0, 0, y - s.offset_y, x - s.offset_x] = 2
        before, expected, wanted = s.TCHW.clone(), s.TCHW.clone(), []
        for name, row in zip(names, rows):
            x, y, value, wx, wy, new = map(int, row[:6])
            if before[0, 0, y - s.offset_y, x - s.offset_x] == value:
                wanted.append(name)
                expected[0, 0, wy - s.offset_y, wx - s.offset_x] = new
        if s.candidate_plan is not None:
            s.candidate_plan.events()
        else:
            s.apply_spatial_events()
        actual = [name for name in names if s.event_history[0][name] == [123]]
        assert actual == wanted and torch.equal(expected, s.TCHW), active
    return {"events": len(rows), "unique_reference_coordinates": len(refs),
            "cases": len(cases), "failures": []}


def run_fixture(base, fixture, steps, seed):
    name, bounds, source, downstream, counter, cjoins = fixture
    x0, _, y0, _ = bounds
    crop = crop_cells(base, bounds)
    for x, y in [source, downstream, counter]:
        assert crop[y - y0, x - x0] == 1, (name, x, y)
    s = copy.copy(base)
    s.trial_ids = list(range(9))
    s.cellspace, s.offset_x, s.offset_y = crop, x0, y0
    # The real counter retains its original 2 -> 1 operation. The downstream
    # sink is a harness boundary, placed beyond the complete monitor circuit.
    s.spatial_event_arrays = np.array([
        [*counter, 2, *counter, 1, 1, -1, -1],
        [*downstream, 2, *downstream, 1, 1, -1, -1],
    ], dtype=float)
    s.spatial_event_names = ["counter", "downstream"]
    s.Allocate_torch_Tensors_on_Device()
    s.set_ParallelTrial(9)
    result = {"name": name, "bounds": bounds, "source": source, "downstream": downstream,
              "counter": counter, "cjoins": cjoins, "cases": []}
    for label, probability, bin_probability in SETTINGS:
        s.TCHW.copy_(torch.from_numpy(crop)[None, None].expand(9, 1, *crop.shape))
        s._current_step = 0
        s.event_history = [{key: [] for key in s.spatial_event_names} for _ in range(9)]
        s.rule_probs_tensor.copy_(s.rule_probs_base_tensor)
        for ri, rid in enumerate(s.rule_ids):
            if rid >= 1000:  # The eight Token bin creation/absorption rules.
                s.rule_probs_tensor[ri] = bin_probability
        hist = np.zeros((9, len(cjoins), 16), int)
        accepted = np.zeros((9, len(cjoins)), int)
        trace, start = [], time.perf_counter()
        for step in range(steps):
            if step < 8:
                s.TCHW[step + 1, 0, source[1] - y0, source[0] - x0] = 2
            state, codes = s.TCHW[:, 0].numpy(), []
            for j, (cx, cy) in enumerate(cjoins):
                code = sum((state[:, cy-y0+dy, cx-x0+dx] == 2) * bit
                           for dx, dy, bit in [(0, -1, 1), (-1, 0, 2), (1, 0, 4), (0, 1, 8)])
                hist[np.arange(9), j, code] += 1
                codes.append(code.tolist())
            if step < 120:
                trace.append(codes)
            s.step(probability, seed=seed)
            for j, (cx, cy) in enumerate(cjoins):
                for rid in [8, 9, 10, 11]:
                    accepted[:, j] += s.TNHW_boolMask[:, s.rule_id_to_index[rid], cy-y0, cx-x0].numpy()
        assert not any(s.event_history[0].values()), "Monitor emitted without an input pulse"
        case = {"label": label, "global_prob": probability, "bin_rule_probability": bin_probability,
                "elapsed_sec": time.perf_counter() - start, "events": s.event_history,
                "cjoin_accepts": accepted.tolist(), "neighborhood_hist": hist.tolist(),
                "first_120_neighborhoods": trace}
        result["cases"].append(case)
        print(name, label, "counter", [len(v["counter"]) for v in s.event_history],
              "downstream", [len(v["downstream"]) for v in s.event_history], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["reference", "torch_sparse"], default="torch_sparse")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--fixture", choices=[v[0] for v in FIXTURES])
    parser.add_argument("--predicates-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 8:
        parser.error("steps must cover all eight pulse injection times")
    torch.set_num_threads(1)
    base = make_base(args.mode)
    result = {"mode": args.mode, "rng": base.rng_mode, "steps": args.steps, "seed": args.seed,
              "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [CELLSPACE, RULES, EVENTS]},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "trials": ["no input"] + [f"one token at step {i}" for i in range(8)],
              "event_predicates": check_predicates(base), "geometry": check_geometry(base), "fixtures": []}
    for fixture in ([] if args.predicates_only else FIXTURES):
        if args.fixture is None or args.fixture == fixture[0]:
            result["fixtures"].append(run_fixture(base, fixture, args.steps, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
