"""Preserve the completed 3M runs and verify their 6M continuations.

The GPU preflight uses isolated copies of the actual 3M cell states. Its empty
history prefixes are verification fixtures only, never production results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil

import torch

from PyBCA.api import Engine
from PyBCA.api.streaming import atomic_json, atomic_write, identity, iter_history, sha256
import check_bca_ip_variants as checks

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "results/variants-20261007"
OUTPUT_ROOT = ROOT / "results/variants-6m-20261013"
SOURCE_STEP = 3_000_000
TARGET_STEP = 6_000_000
VERIFY_STEPS = 128


def stamp():
    return {**checks.source_stamp(), "continuation_script_sha256": sha256(__file__)}


def rank_config(variant, rank, *, device="cpu", steps=TARGET_STEP):
    return replace(checks.config_for(variant, device=device, trials=64, steps=steps,
                   seed=checks.SEED, trial_ids=tuple(range(rank * 64, (rank + 1) * 64))),
                   trial_offset=rank * 64, distributed_mode="off")


def validate_restart(state, config, *, minimum=SOURCE_STEP, maximum=TARGET_STEP):
    if state["manifest"]["identity"] != identity(config):
        raise ValueError("Restart identity differs: inputs, implementation, seed or trial IDs")
    step = state["next_step"]
    if not minimum <= step <= maximum or state["manifest"]["next_step"] != step:
        raise ValueError(f"Unexpected restart step: {step}")
    return step


def prepare(source, output):
    checks.gate(source / "preflight")
    # Exclusive creation prevents recopying over a continuation already running.
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema": "pybca-three-condition-continuation-v1", "passed": False,
              "created_at": datetime.now(timezone.utc).isoformat(), "source": stamp(),
              "source_root": str(source.resolve()), "output_root": str(output.resolve()),
              "source_step": SOURCE_STEP, "target_step": TARGET_STEP,
              "seed": checks.SEED, "trials_per_condition": 512, "variants": []}
    atomic_json(output / "continuation.json", record)
    for variant in checks.VARIANTS:
        src, dst = source / variant, output / variant
        audit = json.loads((src / "completion-audit.json").read_text())
        if not audit["passed"] or audit["source"] != checks.source_stamp():
            raise ValueError(f"Missing or stale 3M completion audit: {variant}")
        if audit["steps"] != SOURCE_STEP or audit["seed"] != checks.SEED or audit["trials"] != 512:
            raise ValueError(f"Unexpected source cohort: {variant}")
        audit_ranks = {row["rank"]: row for row in audit["ranks"]}
        if set(audit_ranks) != set(range(8)):
            raise ValueError("Incomplete source audit rank roster")
        shutil.copytree(src, dst, copy_function=shutil.copy2)
        (dst / "completion-audit.json").rename(dst / "source-3m-completion-audit.json")
        rows = []
        for rank in range(8):
            name = f"rank_{rank:04d}"
            directory = dst / name
            manifest = json.loads((directory / "manifest.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            checks.validate_summary(summary, manifest, variant=variant, rank=rank,
                                    steps=SOURCE_STEP, seed=checks.SEED)
            checkpoint = directory / "checkpoint.pt"
            expected = audit_ranks[rank]["checkpoint_sha256"]
            if sha256(checkpoint) != expected or sha256(src / name / "checkpoint.pt") != expected:
                raise ValueError(f"Changed 3M checkpoint: {variant}/{name}")
            for file in ("checkpoint.pt", "manifest.json", "summary.json"):
                if sha256(directory / file) != sha256(src / name / file):
                    raise ValueError(f"Copy mismatch: {variant}/{name}/{file}")
                if os.path.samefile(directory / file, src / name / file):
                    raise ValueError("Production copies must not share mutable inodes")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            validate_restart(state, rank_config(variant, rank), maximum=SOURCE_STEP)
            if state["manifest"] != manifest:
                raise ValueError("Checkpoint and committed history differ")
            previous = 0
            for chunk in manifest["chunks"]:
                if chunk["start_step"] != previous or chunk["next_step"] <= previous:
                    raise ValueError("History has gaps or overlaps")
                if sha256(directory / chunk["path"]) != chunk["sha256"]:
                    raise ValueError(f"Corrupt copied history: {directory / chunk['path']}")
                previous = chunk["next_step"]
            if previous != SOURCE_STEP:
                raise ValueError("History does not reach 3M")
            rows.append({"rank": rank, "checkpoint_sha256": expected,
                         "history_chunks_verified": len(manifest["chunks"]),
                         "source_manifest_sha256": sha256(src / name / "manifest.json"),
                         "trials": 64, "step": SOURCE_STEP})
            del state
            print(json.dumps({"prepared": variant, **rows[-1]}), flush=True)
        record["variants"].append({"variant": variant, "ranks": rows, "passed": True})
        atomic_json(output / "continuation.json", record)
    record["passed"] = True
    atomic_json(output / "continuation.json", record)


def prepared(output):
    checks.inputs()
    record = json.loads((output / "continuation.json").read_text())
    if (not record["passed"] or record["source"] != stamp() or
            record["source_step"] != SOURCE_STEP or record["target_step"] != TARGET_STEP or
            {r["variant"] for r in record["variants"]} != set(checks.VARIANTS)):
        raise ValueError("Missing, incomplete or stale continuation preparation")
    return record


def verification_checkpoint(source, destination):
    """Keep every state tensor and counter, but omit the old history in a fixture."""
    state = torch.load(source, map_location="cpu", weights_only=True)
    state["manifest"] = copy.deepcopy(state["manifest"])
    state["manifest"]["chunks"] = []
    destination.mkdir(parents=True, exist_ok=False)
    atomic_write(destination / "checkpoint.pt", lambda stream: torch.save(state, stream))
    return state["next_step"]


def snapshot_run(config):
    result = Engine(config).run()
    sim = result.simulator
    snapshot = {"cells": sim.TCHW.cpu().clone(), "rule_probs": sim.rule_probs_tensor.cpu().clone(),
                "step": result.current_step, "offset": (sim.offset_x, sim.offset_y)}
    del result, sim
    gc.collect()
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return snapshot


def records(directory):
    return Counter((row["kind"], row["trial"], row["step"], row["name"], row["count"])
                   for row in iter_history(directory, verify=True))


def parity(output, rank, device):
    record = prepared(output)
    runs = []
    for variant in checks.VARIANTS:
        source = output / variant / f"rank_{rank:04d}/checkpoint.pt"
        expected = next(r for v in record["variants"] if v["variant"] == variant
                        for r in v["ranks"] if r["rank"] == rank)["checkpoint_sha256"]
        if sha256(source) != expected:
            raise ValueError("Preflight requires the unchanged 3M checkpoint")
        scratch = output / "preflight" / f"rank_{rank:04d}" / variant
        full, split = scratch / "uninterrupted", scratch / "split"
        for directory in (full, split):
            if verification_checkpoint(source, directory) != SOURCE_STEP:
                raise ValueError("Verification fixture must start at 3M")
        config = rank_config(variant, rank, device=device, steps=SOURCE_STEP + VERIFY_STEPS)
        config = replace(config, flush_interval=64, checkpoint_interval=64)
        a = snapshot_run(replace(config, stream_dir=str(full), resume_from=str(full / "checkpoint.pt")))
        split_config = replace(config, stream_dir=str(split), resume_from=str(split / "checkpoint.pt"))
        snapshot_run(replace(split_config, steps=SOURCE_STEP + VERIFY_STEPS // 2))
        b = snapshot_run(split_config)
        full_events, split_events = records(full), records(split)
        tests = {"cells": torch.equal(a["cells"], b["cells"]),
                 "rule_probs": torch.equal(a["rule_probs"], b["rule_probs"]),
                 "step": a["step"] == b["step"] == SOURCE_STEP + VERIFY_STEPS,
                 "offset": a["offset"] == b["offset"], "events": full_events == split_events,
                 "source_checkpoint_unchanged": sha256(source) == expected}
        row = {"variant": variant, "rank": rank, "trials": 64, "steps": VERIFY_STEPS,
               "source_checkpoint_sha256": expected, "event_records": sum(full_events.values()),
               "checks": tests, "passed": all(tests.values())}
        runs.append(row)
        atomic_json(output / "preflight" / f"rank_{rank:04d}.json",
                    {"source": stamp(), "runs": runs,
                     "passed": len(runs) == 3 and all(r["passed"] for r in runs)})
        print(json.dumps(row), flush=True)
        if not row["passed"]:
            raise RuntimeError("Restart trajectory differs from uninterrupted trajectory")


def parity_report(output):
    prepared(output)
    rows = [json.loads((output / "preflight" / f"rank_{rank:04d}.json").read_text())
            for rank in range(8)]
    if any(not row["passed"] or row["source"] != stamp() for row in rows):
        raise ValueError("Incomplete GPU restart verification")
    runs = [run for row in rows for run in row["runs"]]
    if {(r["variant"], r["rank"]) for r in runs} != {(v, r) for v in checks.VARIANTS for r in range(8)}:
        raise ValueError("Incomplete GPU verification roster")
    if any(sum(r["event_records"] for r in runs if r["variant"] == v) == 0 for v in checks.VARIANTS):
        raise ValueError("Preflight observed no natural events")
    report = {"passed": True, "source": stamp(), "runs": runs, "trials_checked": 1536,
              "scope": "Actual 3M cells, 128 uninterrupted vs 64+64 resumed GPU updates; isolated verification fixtures."}
    atomic_json(output / "preflight/preflight.json", report)
    print(json.dumps(report), flush=True)


def gate(output, variant):
    record = prepared(output)
    report = json.loads((output / "preflight/preflight.json").read_text())
    if not report["passed"] or report["source"] != stamp():
        raise ValueError("GPU restart verification has not passed for this implementation")
    source_rows = next(r["ranks"] for r in record["variants"] if r["variant"] == variant)
    steps = []
    for row in source_rows:
        path = output / variant / f"rank_{row['rank']:04d}/checkpoint.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        step = validate_restart(state, rank_config(variant, row["rank"]))
        if step == SOURCE_STEP and sha256(path) != row["checkpoint_sha256"]:
            raise ValueError("Changed initial continuation checkpoint")
        steps.append(step)
    print(json.dumps({"variant": variant, "restart_steps": steps, "target_step": TARGET_STEP}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "parity", "parity-report", "gate", "completed"))
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--variant", choices=checks.VARIANTS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        prepare(args.source, args.output)
    elif args.command == "parity":
        rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.device_count() != 8:
            raise RuntimeError("Expected eight allocated GPUs")
        parity(args.output, rank, f"cuda:{rank}")
    elif args.command == "parity-report":
        parity_report(args.output)
    else:
        if args.variant is None:
            parser.error("--variant is required")
        gate(args.output, args.variant)
        if args.command == "completed":
            run = args.output / args.variant
            report = checks.audit_run(run, args.variant, steps=TARGET_STEP, seed=checks.SEED)
            report["continuation_sha256"] = sha256(args.output / "continuation.json")
            atomic_json(run / "completion-audit.json", report)
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
