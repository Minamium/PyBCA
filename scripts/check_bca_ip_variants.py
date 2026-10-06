"""GPU preflight and completion checks for the three paper cellspace variants.

The pilot has a separate seed and output tree. Passing it establishes finite
CPU/CUDA parity and observable circuit activity, not optimization success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re

import numpy as np
import torch

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import atomic_json, identity, iter_history, sha256
from PyBCA.core.io import load_cell_space_yaml_to_numpy

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "Sample/Cellspace/BCA-IP-variants"
VARIANTS = ("condition1_N1", "condition1_N2", "condition2_N2")
SEED = 20261007
PILOT_SEED = 20261008
PILOT_STEPS = 20000
TARGET_STEPS = 3000000
TRIALS = 512
RANKS = 8


def inputs():
    manifest = json.loads((MAPS / "manifest.json").read_text())
    paths = {MAPS / entry["file"]: entry["sha256"] for entry in manifest["variants"]}
    paths[MAPS / manifest["events"]] = manifest["events_sha256"]
    paths[ROOT / manifest["rules"]] = manifest["rules_sha256"]
    if {p.stem.removeprefix("BCA-IP_") for p in paths if p.parent == MAPS and p.suffix == ".yaml"} != set(VARIANTS):
        raise ValueError("Unexpected cellspace roster")
    for path, expected in paths.items():
        if sha256(path) != expected:
            raise ValueError(f"Changed experiment input: {path}")
    for entry in manifest["variants"]:
        instance = json.loads((MAPS / f"condition{entry['condition']}.json").read_text())
        if any(entry[key] != instance[key] for key in ("a", "b", "c")):
            raise ValueError("Readout instance does not match the cellspace manifest")
    return manifest


def config_for(variant, *, device="cpu", trials=2, steps=128, seed=PILOT_SEED,
               trial_ids=(31, 511), stream_dir=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return Config(cellspace_path=str(MAPS / f"BCA-IP_{variant}.yaml"),
                  rule_paths=[str(ROOT / "Sample/rule/base-rule.yaml")],
                  spatial_event_file_path=str(MAPS / "BCA-IP_wide_events.py"),
                  device=device, execution_mode="cuda" if device.startswith("cuda") else "torch_sparse",
                  rng_mode="independent", global_prob=.5, seed=seed, trials=trials,
                  trial_ids=tuple(trial_ids), steps=steps, quiet=True, use_tqdm="false",
                  stream_dir=stream_dir, flush_interval=1000, checkpoint_interval=10000,
                  candidate_capacity=4096, log_level="warning")


def source_stamp():
    sample = identity(config_for(VARIANTS[0]))
    return {"implementation": sample["implementation"], "rng_version": sample["rng_version"],
            "input_manifest_sha256": sha256(MAPS / "manifest.json"),
            "checker_sha256": sha256(Path(__file__)),
            "runner_sha256": sha256(ROOT / "scripts/run_bca_ip.py")}


def parity(output):
    inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("The parity check requires an allocated CUDA GPU")
    rows = []
    for variant in VARIANTS:
        cfg = replace(config_for(variant), record_rule_history=True)
        cpu = Engine(cfg).run().simulator
        gpu = Engine(replace(cfg, device="cuda:0", execution_mode="cuda")).run().simulator
        checks = {name: torch.equal(getattr(cpu, name).cpu(), getattr(gpu, name).cpu())
                  for name in ("TCHW", "TNHW_boolMask", "TCHW_applied", "rule_probs_tensor")}
        checks.update(events=cpu.event_history == gpu.event_history,
                      rules=cpu.rule_history == gpu.rule_history,
                      step=cpu._current_step == gpu._current_step == cfg.steps)
        # Force all observation predicates, including paired reset set/clear
        # events. This tests the shifted reset targets even before natural RSM
        # activity occurs in an integrated trajectory.
        for sim in (cpu, gpu):
            for event in sim.spatial_event_arrays:
                x, y, value = map(int, event[:3])
                sim.TCHW[:, 0, y - sim.offset_y, x - sim.offset_x] = value
        cpu.apply_spatial_events()
        gpu.candidate_plan.events()
        checks["all_event_targets"] = torch.equal(cpu.TCHW, gpu.TCHW.cpu())
        checks["all_event_records"] = cpu.event_history == gpu.event_history
        record = {"variant": variant, "steps": cfg.steps, "trial_ids": list(cfg.trial_ids),
                  "event_predicates": len(cpu.spatial_event_names), "checks": checks,
                  "passed": all(checks.values())}
        rows.append(record)
        atomic_json(output, {"source": source_stamp(), "runs": rows,
                             "passed": len(rows) == len(VARIANTS) and all(r["passed"] for r in rows)})
        print(json.dumps(record), flush=True)
        if not record["passed"]:
            raise RuntimeError(f"CPU/CUDA parity failed: {variant}")
        del cpu, gpu
        gc.collect()
        torch.cuda.empty_cache()


def event_family(name):
    if re.fullmatch(r"[AB]_core_input_[1-6]", name):
        return "TD"
    if re.fullmatch(r"[AB]_x[1-6]output", name):
        return "FSM"
    if re.fullmatch(r"[AB]_Amp_x[1-6]output", name):
        return "Amp"
    if name in ("F_value_A", "F_value_B"):
        return "F"
    if name.startswith("Reset_Signal_") and name.endswith("_set"):
        return "reset"
    return "other"


def validate_summary(summary, manifest, *, variant, rank, steps, seed):
    expected_ids = list(range(rank * 64, (rank + 1) * 64))
    if (summary["current_step"] != steps or summary["target_step"] != steps or
            summary["stopped"] or summary["rank"] != rank or not summary["active"] or
            summary["trial_ids"] != expected_ids or manifest["next_step"] != steps):
        raise ValueError(f"Incomplete run or incorrect trial partition: {variant}, rank {rank}")
    expected = identity(config_for(variant, trials=64, steps=steps, seed=seed,
                                   trial_ids=expected_ids))
    # Distributed partitioning preserves the logical sweep offset even when
    # no parameter sweep is requested. It is part of checkpoint identity.
    expected["trial_offset"] = rank * 64
    if manifest["identity"] != expected:
        raise ValueError(f"Checkpoint identity differs: {variant}, rank {rank}")


def audit_run(run, variant, *, steps, seed):
    manifest_inputs = inputs()
    entry = next(row for row in manifest_inputs["variants"] if row["file"] == f"BCA-IP_{variant}.yaml")
    ox, oy = entry["bounds"][:2]
    initial = load_cell_space_yaml_to_numpy(str(MAPS / entry["file"]), include_offset=False)
    rows, counts, first = [], Counter(), {}
    for rank in range(RANKS):
        directory = run / f"rank_{rank:04d}"
        summary = json.loads((directory / "summary.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        validate_summary(summary, manifest, variant=variant, rank=rank, steps=steps, seed=seed)
        state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["next_step"] != steps or state["manifest"] != manifest or state["offset"] != [ox, oy]:
            raise ValueError(f"Checkpoint/manifest mismatch: {directory}")
        cells = state["cells"].numpy()
        if cells.shape != (64, 1, *initial.shape):
            raise ValueError(f"Wrong checkpoint shape: {cells.shape}")
        for trial in cells[:, 0]:
            if (not np.array_equal(trial != 0, initial != 0) or
                    not np.array_equal(trial == -1, initial == -1) or
                    not np.isin(trial, [-1, 0, 1, 2]).all()):
                raise ValueError(f"Static circuit geometry changed: {directory}")
        allowed_ids = set(manifest["trial_ids"])
        for event in iter_history(directory, verify=True):
            if event["trial"] not in allowed_ids or not 0 <= event["step"] < steps:
                raise ValueError(f"Event outside the run: {event}")
            if event["kind"] == "event":
                family = event_family(event["name"])
                counts[family] += event["count"]
                first[family] = min(first.get(family, steps), event["step"])
        rows.append({"rank": rank, "step": steps, "trials": len(allowed_ids),
                     "elapsed_sec": summary["elapsed_sec"],
                     "checkpoint_sha256": sha256(directory / "checkpoint.pt"),
                     "peak_cuda_bytes": max(c["memory"]["cuda_peak_allocated_bytes"] or 0
                                            for c in manifest["chunks"])})
        del state, cells
    return {"variant": variant, "steps": steps, "seed": seed, "trials": TRIALS,
            "ranks": rows, "events_by_family": dict(counts), "first_event_steps": first,
            "passed": True, "source": source_stamp(),
            "checked_at": datetime.now(timezone.utc).isoformat()}


def pilot_report(root):
    stamp = source_stamp()
    comparison = json.loads((root / "parity.json").read_text())
    rules = json.loads((root / "rules-cuda.json").read_text())
    if not comparison["passed"] or comparison["source"] != stamp or not rules["passed"]:
        raise ValueError("The current source has not passed CPU/CUDA and rule checks")
    results = []
    for variant in VARIANTS:
        result = audit_run(root / variant, variant, steps=PILOT_STEPS, seed=PILOT_SEED)
        if not all(result["events_by_family"].get(k, 0) > 0 for k in ("TD", "FSM", "Amp", "F")):
            raise ValueError(f"Pilot has insufficient observed circuit activity: {variant}")
        per_step = max(row["elapsed_sec"] for row in result["ranks"]) / PILOT_STEPS
        result["projection"] = {"seconds_per_step": per_step,
                                "hours_for_3m": per_step * TARGET_STEPS / 3600,
                                "hours_for_3m_with_10pct_margin": per_step * TARGET_STEPS * 1.1 / 3600,
                                "projection_only": True}
        results.append(result)
    report = {"passed": True, "source": stamp, "variants": results,
              "scope": "Short 8-GPU activity, throughput, checkpoint integrity and finite parity; not optimization statistics."}
    atomic_json(root / "preflight.json", report)
    print(json.dumps(report), flush=True)


def gate(root):
    inputs()
    report = json.loads((root / "preflight.json").read_text())
    if not report["passed"] or report["source"] != source_stamp():
        raise ValueError("Missing, failed or stale preflight")
    if {row["variant"] for row in report["variants"]} != set(VARIANTS):
        raise ValueError("Incomplete preflight roster")
    if not all(row["passed"] for row in report["variants"]):
        raise ValueError("One or more variant pilots failed")
    print("All three variant pilots passed for this exact source.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inputs", "parity", "pilot", "gate", "completed"))
    parser.add_argument("--root", type=Path, default=ROOT / "results/variants-20261007/preflight")
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "inputs":
        print(json.dumps(inputs()), flush=True)
    elif args.command == "parity":
        parity(args.root / "parity.json")
    elif args.command == "pilot":
        pilot_report(args.root)
    elif args.command == "gate":
        gate(args.root)
    elif args.command == "completed":
        if args.variant is None:
            parser.error("completed requires --variant and --root pointing to the run")
        report = audit_run(args.root, args.variant, steps=TARGET_STEPS, seed=SEED)
        atomic_json(args.root / "completion-audit.json", report)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
