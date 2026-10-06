"""Exercise the generated reset wiring coupled to the native FSM C5 circuit.

Populate the reachable Post-Pool sublattice, inject one reset, and count returned
Weight tokens at the Pre-Pool entrance. Other FSM logic is deliberately absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from generate_bca_ip_cellspaces import (
    CORE_OFFSETS, DEFAULT_OUTPUT, ROOT, read_map, sha256, write_map,
)
from validate_bca_ip_cellspaces import static_audit

sys.path.insert(0, str(ROOT / "src"))
from PyBCA.core.simulator import BCA_Simulator


def probe(gain, fixture, maps, scratch, trials, pulses, interval, settle, seed):
    source = read_map(maps / fixture["source_file"])
    offset = CORE_OFFSETS[fixture["core"] - 1]
    cells = {p: v for p, v in fixture["cells"].items() if v}
    for y in range(27, 46):
        for x in range(-40, 9):
            cells[x, y] = source.get((x, y + offset), 0)
    for y in range(-1, 27):
        for x in range(-38, -35):
            cells[x, y] = source.get((x, y + offset), 0)
    keys = {p for p, v in cells.items() if v == 2 and p[0] >= -40}
    assert keys == {(-38, 28), (-35, 31)}, keys
    # The filled pool uses two-cell hops. Only this parity connects to C5;
    # arbitrary placement in another sublattice would give a false reset failure.
    post_positions = [(-29 + 2 * i, 37) for i in range(gain)]
    assert all(cells[p] == 1 for p in post_positions)
    folder = scratch / f"gain-{gain}"
    folder.mkdir(parents=True, exist_ok=True)
    cellfile, eventfile = folder / "cellspace.yaml", folder / "events.py"
    write_map(cellfile, cells)
    eventfile.write_text("events = [('pre_pool_return', (-37,0), 2, (-37,0), 1)]\n")
    sim = BCA_Simulator(str(cellfile), [str(ROOT / "Sample/rule/base-rule.yaml")],
                        device="cpu", spatial_event_filePath=str(eventfile), quiet=True,
                        execution_mode="torch_sparse", rng_mode="independent",
                        trial_ids=list(range(trials + 1)))
    sim.Allocate_torch_Tensors_on_Device()
    sim.set_ParallelTrial(trials + 1)
    iy, ix = 36 - sim.offset_y, -113 - sim.offset_x
    post_y = 37 - sim.offset_y
    post_x = [x - sim.offset_x for x, _ in post_positions]
    sim.TCHW[-1, 0, post_y, post_x] = 2  # Same Post-Pool in the no-reset control.
    steps = (pulses - 1) * interval + settle
    start = time.monotonic()
    checkpoints = []
    for step in range(steps):
        if step % interval == 0 and step // interval < pulses:
            returns = [len(h["pre_pool_return"]) for h in sim.event_history]
            assert returns == [gain * (step // interval)] * trials + [0], returns
            assert bool(torch.all(sim.TCHW[:trials, 0, post_y, post_x] == 1)), "Previous Post-Pool not empty"
            assert bool(torch.all(sim.TCHW[:trials, 0, iy, ix] == 1)), "Reset input busy"
            sim.TCHW[:trials, 0, post_y, post_x] = 2
            sim.TCHW[:trials, 0, iy, ix] = 2
            checkpoints.append({"step": step, "returned_before_reload": returns})
        sim.step(.5, seed=seed)
    returns = [h["pre_pool_return"] for h in sim.event_history]
    counts = list(map(len, returns))
    expected = [gain * pulses] * trials + [0]
    post_counts = (sim.TCHW[:, 0, 34-sim.offset_y:46-sim.offset_y,
                            -32-sim.offset_x:9-sim.offset_x] == 2).flatten(1).sum(1).tolist()
    # Weight tokens can leave the lower pool and wait on the C5 input rail at
    # y=30 before a reset. Include that rail and subtract its single ratchet key.
    waiting_counts = ((sim.TCHW[:, 0, 29-sim.offset_y:46-sim.offset_y,
                               -36-sim.offset_x:9-sim.offset_x] == 2)
                      .flatten(1).sum(1) - 1).tolist()
    passed = counts == expected and waiting_counts == [0] * trials + [gain]
    result = {"gain": gain, "source_file": fixture["source_file"],
              "source_core": fixture["core"], "source_sha256": sha256(maps / fixture["source_file"]),
              "fixture_sha256": sha256(cellfile), "post_pool_initial_positions": post_positions,
              "positive_trials": trials, "no_reset_controls": 1, "pulses_per_positive_trial": pulses,
              "pulse_interval": interval, "steps": steps, "seed": seed, "global_prob": .5,
              "returned_Weight_expected": expected, "returned_Weight_observed": counts,
              "lower_Post_Pool_tokens": post_counts,
              "remaining_Weights_in_Post_Pool_and_C5_input": waiting_counts,
              "waiting_region": {"bounds_inclusive": [-36, 29, 8, 45], "ratchet_keys_subtracted": 1},
              "pre_pool_return_steps": returns,
              "between_pulses": checkpoints, "passed": passed,
              "elapsed_sec": time.monotonic() - start}
    np.savez_compressed(folder / "final.npz", cells=sim.TCHW.cpu().numpy(),
                        offset=np.array([sim.offset_x, sim.offset_y]))
    (folder / "result.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ("gain", "returned_Weight_observed", "remaining_Weights_in_Post_Pool_and_C5_input", "passed")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "weight-reset-validation.json")
    parser.add_argument("--scratch", type=Path, default=ROOT / "results/cellspace-variants-20261006/weight-reset")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--pulses", type=int, default=2)
    parser.add_argument("--interval", type=int, default=6000)
    parser.add_argument("--settle", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    if min(args.trials, args.pulses, args.interval, args.settle) < 1:
        parser.error("Probe dimensions must be positive")
    torch.set_num_threads(1)
    _, fixtures = static_audit(args.maps)
    report = {"schema_version": 1, "script_sha256": sha256(Path(__file__)),
              "definition": "Generated reset Amp + native C5 + Post-Pool; absorbing Pre-Pool entrance.",
              "limits": "This verifies Weight return through C5, not full FSM Key/Ans/TD recovery or long-run search.",
              "probes": []}
    for gain, fixture in sorted(fixtures.items()):
        report["probes"].append(probe(gain, fixture, args.maps, args.scratch,
            args.trials, args.pulses, args.interval, args.settle, args.seed))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    report["passed"] = all(p["passed"] for p in report["probes"])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not report["passed"]:
        raise SystemExit("Weight reset validation failed")


if __name__ == "__main__":
    main()
