"""Measure the largest per-GPU BCA-IP batch fitting a declared time/memory budget.

This probes simulation and event-history recording only. A production run with
an additional decoder must include its overhead in a final calibration before
using this provisional recommendation. Benchmark trials are never statistics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from PyBCA.api.streaming import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def assess_measurement(record, *, target_steps, wall_seconds, reserve_seconds,
                       slowdown_factor, memory_limit_bytes, checkpoint_interval):
    """Use the slowest observed repeat, including amortized checkpoint cost."""
    seconds_per_step = max(record["seconds_per_step_samples"])
    checkpoint_seconds = record["flush_checkpoint_sec"]
    # The timed repeats already include periodic event-history flushes.
    projected = (target_steps * (seconds_per_step + checkpoint_seconds/checkpoint_interval)
                 * slowdown_factor + record["setup_sec"] + checkpoint_seconds)
    peak = max(record["peak_allocated_bytes"], record["peak_reserved_bytes"])
    return {"projected_run_seconds": projected, "reserved_seconds": reserve_seconds,
            "projected_total_seconds": projected + reserve_seconds,
            "seconds_per_step_upper": seconds_per_step, "peak_memory_bytes": peak,
            "time_fits": projected + reserve_seconds <= wall_seconds,
            "memory_fits": peak <= memory_limit_bytes,
            "fits": projected + reserve_seconds <= wall_seconds and peak <= memory_limit_bytes}


def largest_measured_batch(measure, *, initial=32, maximum=2048):
    """Bracket then bisect; return only a batch actually measured as feasible.

    Search assumes approximately monotonic resource cost with batch size. All
    probe results remain available for checking that assumption afterwards.
    """
    if initial < 1 or maximum < initial:
        raise ValueError("Require 1 <= initial <= maximum")
    low, high = 0, None
    n = initial
    while True:
        fits = measure(n)
        if fits:
            low = n
            if n == maximum:
                return low
            if high is None:
                n = min(maximum, n*2)
                continue
        else:
            high = n
        if high-low <= 1:
            return low
        n = (low+high)//2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, default=3_000_000)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--wall-hours", type=float, default=24.)
    parser.add_argument("--reserve-seconds", type=float, default=3600.)
    parser.add_argument("--slowdown-factor", type=float, default=1.10)
    parser.add_argument("--memory-fraction", type=float, default=.90)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--steps-per-repeat", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--initial-trials", type=int, default=32)
    parser.add_argument("--maximum-trials", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not (args.target_steps > 0 and args.gpus > 0 and args.checkpoint_interval > 0
            and args.steps_per_repeat >= 1000 and args.repeats >= 2
            and 0 < args.memory_fraction < 1 and args.slowdown_factor >= 1
            and 0 <= args.reserve_seconds < args.wall_hours*3600):
        parser.error("Invalid budget; use >=1000 steps/repeat and >=2 repeats")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    memory_limit = int(props.total_memory*args.memory_fraction)
    inputs = [ROOT/"Sample/Cellspace/BCA-IP.yaml", ROOT/"Sample/rule/base-rule.yaml",
              ROOT/"Sample/Specialevent/BCA-IP_event.py"]
    sources = [p for folder in ("core", "api") for p in (ROOT/"src/PyBCA"/folder).rglob("*")
               if p.suffix in {".py", ".cu"}]
    sources += [Path(__file__), ROOT/"scripts/benchmark_core.py"]
    plan = {"schema": "pybca-capacity-v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "target_steps": args.target_steps, "gpus_requested_for_production": args.gpus,
            "gpus_used_for_this_probe": 1, "wall_seconds": args.wall_hours*3600,
            "reserve_seconds": args.reserve_seconds, "slowdown_factor": args.slowdown_factor,
            "checkpoint_interval": args.checkpoint_interval, "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory, "memory_limit_bytes": memory_limit,
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(inputs+sources)}, "measurements": [],
            "production_ready": False,
            "pending": ["confirmed optimum decoder", "8-GPU calibration including decoder overhead"],
            "completed": False}
    atomic_json(out/"plan.json", plan)

    def measure(trials):
        print(f"Probing {trials} trials on {args.device}", flush=True)
        result_path = out/f"trials-{trials:04d}.json"
        command = [sys.executable, str(ROOT/"scripts/benchmark_core.py"),
                   "--mode", "cuda", "--rng", "independent", "--device", args.device,
                   "--trials", str(trials), "--steps", str(args.steps_per_repeat),
                   "--repeats", str(args.repeats), "--output", str(result_path)]
        # Separate processes free each candidate's allocator/context completely.
        with (out/f"trials-{trials:04d}.txt").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            log_text = (out/f"trials-{trials:04d}.txt").read_text()
            if "out of memory" not in log_text.lower():
                raise RuntimeError(f"Probe {trials} failed; see {out}/trials-{trials:04d}.txt")
            row = {"trials": trials, "fits": False, "reason": "out_of_memory"}
        else:
            record = json.loads(result_path.read_text())
            row = {"trials": trials, **assess_measurement(record, target_steps=args.target_steps,
                    wall_seconds=args.wall_hours*3600, reserve_seconds=args.reserve_seconds,
                    slowdown_factor=args.slowdown_factor, memory_limit_bytes=memory_limit,
                    checkpoint_interval=args.checkpoint_interval)}
        plan["measurements"].append(row)
        atomic_json(out/"plan.json", plan)
        print(json.dumps(row), flush=True)
        return row["fits"]

    chosen = largest_measured_batch(measure, initial=args.initial_trials, maximum=args.maximum_trials)
    plan["measured_recommendation_per_gpu"] = chosen
    plan["provisional_total_trials"] = chosen*args.gpus
    plan["next_batch_size_is_infeasible"] = any(r["trials"] == chosen+1 and not r["fits"]
                                                   for r in plan["measurements"])
    plan["completed"] = True
    atomic_json(out/"plan.json", plan)
    print(json.dumps({k: plan[k] for k in ("measured_recommendation_per_gpu", "provisional_total_trials",
                                          "production_ready", "pending")}), flush=True)
    if not chosen:
        raise RuntimeError("Even one trial did not fit the declared budget")


if __name__ == "__main__":
    main()
