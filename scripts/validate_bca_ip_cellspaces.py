"""Validate generated parameters, changed regions, ports and reset pulse counts.

The dynamic test cuts the actual reset wiring out of the generated YAML files.
It does not certify whole-FSM restoration or long-run optimization statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

from generate_bca_ip_cellspaces import (
    BASE, CORE_OFFSETS, DEFAULT_OUTPUT, EVENTS, PATCH, ROOT, condition1_source,
    read_map, sha256, shifted_x, widen, write_map,
)

sys.path.insert(0, str(ROOT / "src"))
from PyBCA.core.simulator import BCA_Simulator


def static_audit(folder: Path) -> tuple[dict, dict]:
    manifest = json.loads((folder / "manifest.json").read_text())
    base = read_map(BASE)
    patch = json.loads(PATCH.read_text())
    sources = {1: condition1_source(base, patch), 2: base}
    assert sha256(BASE) == manifest["base_sha256"] == patch["base_sha256"]
    assert sha256(PATCH) == manifest["condition1_patch_sha256"]
    events = runpy.run_path(str(folder / manifest["events"]))["events"]
    original_events = runpy.run_path(str(EVENTS))["events"]
    assert len(events) == len(original_events)
    for actual, original in zip(events, original_events):
        name, ref, state, dst, value = original
        assert actual == (name, (shifted_x(ref[0]), ref[1]), state,
                          (shifted_x(dst[0]), dst[1]), value)
    assert sha256(folder / manifest["events"]) == manifest["events_sha256"]
    report, fixtures = [], {}
    maps = {}
    for entry in manifest["variants"]:
        path = folder / entry["file"]
        cells = read_map(path)
        maps[entry["file"]] = cells
        assert sha256(path) == entry["sha256"]
        expected_w = [entry["N"] * (max(entry["c"]) - c) + 1 for c in entry["c"]]
        assert expected_w == entry["weights"] == entry["reset_gains"]
        baseline = widen(sources[entry["condition"]])
        changed = {p for p in baseline.keys() | cells.keys() if baseline.get(p, 0) != cells.get(p, 0)}
        permitted = {(x, o - 2) for o in CORE_OFFSETS for x in range(-61, -52)}
        permitted |= {(x, o + y) for o in CORE_OFFSETS
                      for y in range(3, 43) for x in range(-113, -40)}
        assert not changed - permitted, sorted(changed - permitted)[:20]
        for event in events:
            assert cells.get(event[1], 0) in (1, 2), event
            assert cells.get(event[3], 0) in (1, 2), event
        counts = []
        for i, o in enumerate(CORE_OFFSETS):
            w = expected_w[i % 6]
            observed = sum(cells.get((x, o - 2), 0) == 2 for x in range(-61, -52))
            assert observed == w
            counts.append(observed)
            region = {(x, y): cells.get((x, o + y), 0)
                      for y in range(3, 43) for x in range(-113, -40)}
            if w in fixtures:
                assert region == fixtures[w]["cells"], (entry["file"], i, w)
            else:
                fixtures[w] = {"cells": region, "source_file": entry["file"], "core": i + 1}
        capacities = []
        for o in (0, 495):
            b = sum(cells.get((shifted_x(x), 217 + o), 0) == 2 for x in range(-280, -258))
            assert b == entry["b"]
            capacities.append(b)
        report.append({"file": entry["file"], "sha256": entry["sha256"],
                       "weight_counts_A_B": counts, "TD_capacity_A_B": capacities,
                       "changed_cells_inside_allowed_regions": len(changed),
                       "outside_allowed_regions_changed": 0,
                       "event_coordinates_valid": True})
    # N only changes the weights/reset circuits for the matched condition-1 pair.
    a, b = (maps[f"BCA-IP_condition1_N{n}.yaml"] for n in (1, 2))
    difference = {p for p in a.keys() | b.keys() if a.get(p, 0) != b.get(p, 0)}
    assert not difference - permitted
    legacy_weights = [sum(base.get((x, o - 2), 0) == 2 for x in range(-61, -52))
                      for o in CORE_OFFSETS[:6]]
    legacy_gains = [1 + sum(base.get((x, o + y), 0) == -1
                           for x in range(-64, -53) for y in range(6, 40))
                    for o in CORE_OFFSETS[:6]]
    return {"variants": report, "matched_N1_N2_changed_cells": len(difference),
            "legacy_condition2_initial_weights": legacy_weights,
            "legacy_reset_ladder_gains_by_stage_count": legacy_gains}, fixtures


def probe(gain: int, fixture: dict, scratch: Path, trials: int, pulses: int,
          interval: int, settle: int, seed: int) -> dict:
    folder = scratch / f"gain-{gain}"
    folder.mkdir(parents=True, exist_ok=True)
    cells = {p: v for p, v in fixture["cells"].items() if v}
    flag = (-115, 0)
    cells[flag] = 1
    cellfile = folder / "cellspace.yaml"
    write_map(cellfile, cells)
    input_port, output_port = (-113, 36), (-41, 35)
    events = [(f"inject_{p}", flag, 1, input_port, 2, 1., p * interval, p * interval)
              for p in range(pulses)]
    events.append(("output", output_port, 2, output_port, 1))
    eventfile = folder / "events.py"
    eventfile.write_text(f"events = {events!r}\n")
    sim = BCA_Simulator(str(cellfile), [str(ROOT / "Sample/rule/base-rule.yaml")],
                        device="cpu", spatial_event_filePath=str(eventfile), quiet=True,
                        execution_mode="torch_sparse", rng_mode="independent",
                        trial_ids=list(range(trials + 1)))
    sim.Allocate_torch_Tensors_on_Device()
    sim.set_ParallelTrial(trials + 1)
    # Final trial is a no-input control. Its isolated flag cannot move under CA.
    sim.TCHW[-1, 0, flag[1] - sim.offset_y, flag[0] - sim.offset_x] = 2
    start = time.monotonic()
    steps = (pulses - 1) * interval + settle
    for step in range(steps):
        if step % interval == 0 and step // interval < pulses:
            assert bool(torch.all(sim.TCHW[:trials, 0,
                input_port[1] - sim.offset_y, input_port[0] - sim.offset_x] == 1)), "Input was busy"
        sim.step(.5, seed=seed)
    counts = [len(row["output"]) for row in sim.event_history]
    expected = [gain * pulses] * trials + [0]
    injections = [sum(len(row[f"inject_{p}"]) for p in range(pulses)) for row in sim.event_history]
    passed = counts == expected and injections == [pulses] * trials + [0]
    np.savez_compressed(folder / "final.npz", cells=sim.TCHW.cpu().numpy(),
                        offset=np.array([sim.offset_x, sim.offset_y]))
    result = {"gain": gain, "source_file": fixture["source_file"],
              "source_core": fixture["core"], "fixture_sha256": sha256(cellfile),
              "steps": steps, "independent_positive_trials": trials, "no_input_trials": 1,
              "pulses_per_positive_trial": pulses, "pulse_interval": interval,
              "seed": seed, "global_prob": .5, "injection_counts": injections,
              "expected_output_counts": expected, "observed_output_counts": counts,
              "output_steps": [row["output"] for row in sim.event_history],
              "passed": passed, "elapsed_sec": time.monotonic() - start}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ("gain", "observed_output_counts", "passed", "elapsed_sec")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "validation.json")
    parser.add_argument("--scratch", type=Path, default=ROOT / "results/cellspace-variants-20261006/validation")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--pulses", type=int, default=3)
    parser.add_argument("--interval", type=int, default=5000)
    parser.add_argument("--settle", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    for value in (args.trials, args.pulses, args.interval, args.settle):
        if value < 1:
            parser.error("Probe dimensions must be positive")
    torch.set_num_threads(1)
    report, fixtures = static_audit(args.maps)
    report["schema_version"] = 1
    report["script_sha256"] = sha256(Path(__file__))
    report["generator_sha256"] = sha256(ROOT / "scripts/generate_bca_ip_cellspaces.py")
    report["limits"] = ["Reset Amp counts are measured with an absorbing output port.",
                         "Whole-FSM Weight/Key/Ans restoration and long-run optimization remain unverified.",
                         "These maps have a wider layout than the historical 512-trial run."]
    report["dynamic_reset_probes"] = []
    if not args.static_only:
        for gain, fixture in sorted(fixtures.items()):
            report["dynamic_reset_probes"].append(probe(gain, fixture, args.scratch,
                args.trials, args.pulses, args.interval, args.settle, args.seed))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    report["passed"] = all(p["passed"] for p in report["dynamic_reset_probes"])
    report["dynamic_validation_executed"] = not args.static_only
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report["passed"]:
        raise SystemExit("Reset pulse validation failed; see the result JSON")


if __name__ == "__main__":
    main()
