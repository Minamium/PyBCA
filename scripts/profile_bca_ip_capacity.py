"""Continue a completed memory search with wall-time-balanced throughput repeats.

The input plan retains the original 512-trial measurement and capacity probes.
Only the near-capacity cohort is timed again. No simulation code is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from benchmark_bca_ip_gpu import ROOT, measure_cohort
from PyBCA.api.streaming import atomic_json


def validate_source(plan, total_memory):
    if plan.get("schema") != "pybca-single-gpu-capacity-v1":
        raise ValueError("Unrecognized benchmark source plan")
    if plan["gpu_total_memory_bytes"] != total_memory:
        raise ValueError("GPU memory differs from the capacity search")
    chosen = plan.get("largest_short_probe_fitting_memory")
    baseline = plan.get("profile_512", {})
    if not chosen or baseline.get("status") != "ok":
        raise ValueError("Source must contain a completed baseline and memory search")
    for name, expected in plan["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed since capacity search: {name}")
    probes = [row for row in plan["measurements"]
              if row["trials"] == chosen and row.get("memory_fits")]
    if not probes:
        raise ValueError("Chosen capacity has no successful measured probe")
    return chosen, probes[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seconds-per-repeat", type=float, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.seconds_per_repeat < 60 or args.repeats < 3:
        parser.error("Use at least 60 seconds and three repetitions")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Request exactly one GPU")
    raw = args.source_plan.read_bytes()
    plan = json.loads(raw)
    props = torch.cuda.get_device_properties(0)
    if props.name != plan["gpu"]:
        raise ValueError("GPU model differs from the capacity search")
    chosen, probe = validate_source(plan, props.total_memory)
    for name in ["global_prob", "seed", "target_steps", "slowdown_factor", "memory_limit_bytes"]:
        setattr(args, name, plan[name])
    args.probe_timeout = 7200
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan["completed"] = False
    plan["continuation"] = {
        "source_plan": str(args.source_plan.resolve()),
        "source_plan_sha256": hashlib.sha256(raw).hexdigest(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "target_seconds_per_repeat": args.seconds_per_repeat,
        "reason": "Match repeat wall time to the 512-trial profile as cohort updates get slower.",
    }
    atomic_json(out / "plan.json", plan)
    per_step = statistics.median(probe["seconds_per_step_samples"])
    steps = max(64, math.ceil(args.seconds_per_repeat / per_step))
    while chosen:
        record = measure_cohort(args, out, chosen, plan["native_process_trial_limit"],
            label=f"profile-memory-{chosen}", steps=steps, repeats=args.repeats, warmup=100)
        plan["measurements"].append(record)
        atomic_json(out / "plan.json", plan)
        if record["memory_fits"]:
            plan["profile_near_full_memory"] = record
            break
        chosen -= plan["batch_quantum"]
    plan["recommended_memory_cohort"] = chosen
    plan["completed"] = True
    plan["continuation"]["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(out / "plan.json", plan)
    if not chosen:
        raise RuntimeError("No measured cohort fit the memory budget")


if __name__ == "__main__":
    main()
