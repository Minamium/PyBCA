"""Inspect archived BCA-IP grids and replay copies without modifying a run.

The replay keeps the complete grid, original seed, absolute step, trial ID,
rule probabilities and events. It uses the verified CPU independent backend.
These short continuations are diagnostics, not completed statistical trials.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from PyBCA.core.io import extract_cellspace_and_offset, load_cell_space_yaml_to_numpy
from PyBCA.core.simulator import BCA_Simulator

ROOT = Path(__file__).resolve().parents[1]
CELLSPACE = ROOT / "Sample/Cellspace/BCA-IP.yaml"
RULES = ROOT / "Sample/rule/base-rule.yaml"
EVENTS = ROOT / "Sample/Specialevent/BCA-IP_event.py"
GROUPS = ["td_to_fsm", "fsm_to_amp", "amp_output", "unit_output", "comparator", "reset"]
# Inclusive geometric views, not a certified solution decoder.
REGIONS = {
    "all": (-275, 459, -16, 894),
    "A": (-275, 459, -16, 388),
    "B": (-275, 459, 479, 894),
    "A4": (-112, 200, 183, 249),
    "B3": (-112, 200, 612, 678),
    "A4_input": (-106, -80, 196, 218),
    "A4_core": (-60, 65, 183, 248),
    "A4_output": (165, 195, 197, 222),
    "A4_amp": (120, 459, 183, 249),
    "comparator_rsm": (-110, 128, 377, 465),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group(name):
    if "_core_input_" in name:
        return "td_to_fsm"
    if "_Amp_x" in name:
        return "amp_output"
    if name.startswith(("A_x", "B_x")):
        return "fsm_to_amp"
    if name.startswith("F_value"):
        return "unit_output"
    if name.startswith("Comparate"):
        return "comparator"
    return "reset"


def history(directory, manifest):
    totals = Counter({g: 0 for g in GROUPS})
    per_trial = {i: Counter({g: 0 for g in GROUPS}) for i in manifest["trial_ids"]}
    previous = 0
    records = 0
    seen = set()
    for c in manifest["chunks"]:
        raw = (directory / c["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["sha256"], c["path"]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert c["start_step"] == previous
        header = rows[0]["__chunk__"]
        assert header["start_step"] == previous and header["next_step"] == c["next_step"]
        assert len(rows) - 1 == c["records"]
        for r in rows[1:]:
            assert r["kind"] == "event" and r["trial"] in per_trial
            assert previous <= r["step"] < c["next_step"] and r["name"] in manifest["events"]
            key = (r["trial"], r["step"], r["name"])
            assert key not in seen
            seen.add(key)
            totals[group(r["name"])] += r["count"]
            per_trial[r["trial"]][group(r["name"])] += r["count"]
            records += 1
        previous = c["next_step"]
    assert previous == manifest["next_step"]
    return dict(totals), per_trial, records


def check_source(state):
    identity = state["manifest"]["identity"]
    assert identity["inputs"] == {
        "cellspace": digest(CELLSPACE), "rules": [digest(RULES)], "events": digest(EVENTS)
    }
    for relative, sha in identity["implementation"].items():
        assert digest(ROOT / "src/PyBCA" / relative) == sha, relative
    assert identity["rng_mode"] == "independent" and not identity["state_gate_enable"]


def encode_rows(rows):
    """Lossless little-endian token bits, XOR against the preceding frame."""
    a = np.asarray(rows, dtype=np.uint8)
    delta = a.copy()
    delta[1:] = a[1:] ^ a[:-1]
    compressed = gzip.compress(delta.tobytes(), compresslevel=9, mtime=0)
    # Verify the exact transform used by the browser, including byte padding.
    restored = np.frombuffer(gzip.decompress(compressed), dtype=np.uint8).reshape(a.shape)
    assert np.array_equal(np.bitwise_xor.accumulate(restored, axis=0), a)
    return base64.b64encode(compressed).decode("ascii")


def replay(path, trial_id, steps, initial, wire, offset):
    state = torch.load(path, map_location="cpu", weights_only=True)
    check_source(state)
    ids = state["manifest"]["trial_ids"]
    local = ids.index(trial_id)
    identity = state["manifest"]["identity"]
    sim = BCA_Simulator(str(CELLSPACE), [str(RULES)], device="cpu", quiet=True,
                        spatial_event_filePath=str(EVENTS), execution_mode="torch_sparse",
                        rng_mode="independent", trial_ids=[trial_id])
    sim.Allocate_torch_Tensors_on_Device()
    sim.set_ParallelTrial(1)
    sim.TCHW.copy_(state["cells"][local:local+1])
    sim._current_step = state["next_step"]
    probs = state["rule_probs"]
    sim.rule_probs_tensor.copy_(probs if probs.ndim == 1 else probs[local])
    ox, oy = offset
    frames, changes, region_changes, accepts = [], [], {}, Counter()
    changed_any = np.zeros_like(initial, dtype=bool)
    masks = {}
    for key, (x0, x1, y0, y1) in REGIONS.items():
        masks[key] = (slice(y0-oy, y1-oy+1), slice(x0-ox, x1-ox+1))
        region_changes[key] = 0
    start = time.perf_counter()
    for i in range(steps + 1):
        a = sim.TCHW[0, 0].numpy().copy()
        frames.append(np.packbits(a.ravel()[wire] == 2, bitorder="little"))
        if i == steps:
            break
        sim.step(identity["global_prob"], seed=identity["seed"])
        b = sim.TCHW[0, 0].numpy()
        assert np.array_equal(b == 0, initial == 0) and np.array_equal(b == -1, initial == -1)
        changed = a != b
        changed_any |= changed
        changes.append(int(changed.sum()))
        for key, sl in masks.items():
            region_changes[key] += int(changed[sl].sum())
        for ri, rid in enumerate(sim.rule_ids):
            if 8 <= rid <= 11:
                accepts[int(rid)] += int(sim.TNHW_boolMask[:, ri].sum())
    events = {n: v for n, v in sim.event_history[0].items() if v}
    metadata = {
        "source": str(path), "source_sha256": digest(path), "trial_id": trial_id,
        "source_step": state["next_step"], "last_step": sim._current_step,
        "global_prob": identity["global_prob"], "seed": identity["seed"],
        "steps": steps, "elapsed_sec": time.perf_counter()-start,
        "mode": "torch_sparse", "device": "cpu", "rng": "independent",
        "changed_cells_per_step": changes, "unique_changed_cells": int(changed_any.sum()),
        "region_cell_changes_summed_over_steps": region_changes,
        "accepted_cjoin_rules_entire_grid": dict(accepts), "events": events,
        "region_unique_changed_cells": {k: int(changed_any[sl].sum()) for k, sl in masks.items()},
    }
    print(json.dumps({k: metadata[k] for k in ("global_prob", "steps", "elapsed_sec", "unique_changed_cells", "events")}), flush=True)
    return np.array(frames), metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replay-steps", type=int, default=128)
    p.add_argument("--trial", type=int, default=1, choices=(0, 1))
    args = p.parse_args()
    torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve()
    initial, ox, oy = extract_cellspace_and_offset(load_cell_space_yaml_to_numpy(str(CELLSPACE)))
    wire = initial.ravel() > 0
    stride = (int(wire.sum()) + 7) // 8
    expected = json.loads((ROOT / "docs/experiments/2026-09-25-probability-check/stopped-checkpoints.json").read_text())["checkpoints"]
    packed, trials, audits, selected = [], [], [], {}
    directories = sorted((source / "bca-ip-512-trials-20260924").glob("rank_????"))
    assert len(directories) == 8
    directories += [source / "probe-global-prob-0.5-20260925"]
    totals = {"p1": Counter(), "p05": Counter()}
    for index, directory in enumerate(directories):
        path = directory / "checkpoint.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        check_source(state)
        assert digest(path) == expected[index]["checkpoint_sha256"]
        assert state["manifest"] == json.loads((directory / "manifest.json").read_text())
        assert state["next_step"] == state["manifest"]["next_step"]
        assert state["offset"] == [ox, oy]
        cells = state["cells"][:, 0].numpy()
        assert np.isin(cells, [-1, 0, 1, 2]).all()
        assert np.all((cells == 0) == (initial == 0)) and np.all((cells == -1) == (initial == -1))
        counts, per_trial, records = history(directory, state["manifest"])
        label = "p1" if index < 8 else "p05"
        totals[label].update(counts)
        ids = state["manifest"]["trial_ids"]
        assert ids == (list(range(index*64, (index+1)*64)) if index < 8 else [0, 1])
        packed.extend(np.packbits(cells.reshape(len(cells), -1)[:, wire] == 2, axis=1, bitorder="little"))
        for local, trial in enumerate(ids):
            trials.append({"trial": trial, "p": state["manifest"]["identity"]["global_prob"],
                           "step": state["next_step"], "tokens": int((cells[local] == 2).sum()),
                           "events": [per_trial[trial][g] for g in GROUPS]})
            if trial == args.trial:
                selected[label] = cells[local].copy()
        audits.append({"path": str(path), "sha256": digest(path), "shape": list(state["cells"].shape),
                       "step": state["next_step"], "records": records, "chunks": len(state["manifest"]["chunks"]),
                       "counts": counts, "source_and_topology_checks_passed": True})
    packed = np.array(packed)
    np.savez_compressed(out / "selected-states.npz", initial=initial, p1=selected["p1"], p05=selected["p05"], offset=[ox, oy])
    print("Verified 514 grids and all history chunks", flush=True)
    # Shared, exact static topology plus token bits; no display downsampling.
    geometry = np.where(initial == -1, 255, np.where(initial > 0, 1, 0)).astype(np.uint8)
    data = {"shape": list(initial.shape), "offset": [ox, oy], "stride": stride,
            "geometry": base64.b64encode(gzip.compress(geometry.tobytes(), 9, mtime=0)).decode("ascii"),
            "initial": encode_rows([np.packbits(initial.ravel()[wire] == 2, bitorder="little")]),
            "finals": encode_rows(packed), "trials": trials, "regions": REGIONS,
            "groups": GROUPS, "replays": []}
    metadata = []
    for label, directory in (("p1", directories[0]), ("p05", directories[-1])):
        frames, meta = replay(directory / "checkpoint.pt", args.trial, args.replay_steps, initial, wire, [ox, oy])
        np.savez_compressed(out / f"replay-{label}.npz", tokens=frames, offset=[ox, oy], shape=initial.shape)
        data["replays"].append({"label": label, "trial": args.trial, "step": meta["source_step"],
                                "p": meta["global_prob"], "frames": len(frames), "bits": encode_rows(frames)})
        metadata.append(meta)
    summary = {"download": json.loads((source / "download.json").read_text()), "audits": audits,
               "totals": {k: dict(v) for k, v in totals.items()}, "replays": metadata,
               "initial_counts": {str(v): int((initial == v).sum()) for v in (-1, 0, 1, 2)},
               "note": "Geometric token views are not a certified candidate or optimum decoder. Diagnostic continuations do not alter archived runs."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "viewer-data.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(json.dumps({"viewer_data_bytes": (out / "viewer-data.json").stat().st_size,
                      "summary": str(out / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
