"""Measure 512-trial throughput and a near-full single-GPU BCA-IP cohort.

The simulation core is unchanged. Cohorts beyond its per-process indexing
limit use independent trial-ID partitions in separate processes on ONE GPU.
All workers remain resident together; cohort timings span a shared start to
the last worker's synchronized finish. Checkpoints use the output filesystem.
Short capacity probes and longer throughput profiles are labelled separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import HistoryWriter, atomic_json
from PyBCA.core.io import load_cell_space_yaml_to_numpy, load_transition_rules_yaml
from plan_bca_ip_capacity import largest_measured_batch

ROOT = Path(__file__).resolve().parents[1]


def partition_trials(total, native_limit, max_workers=2):
    if total < 1 or native_limit < 1:
        raise ValueError("Positive trial counts required")
    workers = math.ceil(total / native_limit)
    if workers > max_workers:
        raise ValueError("Cohort exceeds the allowed number of GPU-sharing workers")
    sizes = [total // workers + int(i < total % workers) for i in range(workers)]
    start = 0
    result = []
    for count in sizes:
        result.append({"start": start, "count": count})
        start += count
    return result


def project_runtime(samples, checkpoint_sec, setup_sec, *, trials,
                    target_steps=3_000_000, checkpoint_interval=10_000,
                    slowdown=1.10):
    if not samples or any(s <= 0 for s in samples) or trials < 1:
        raise ValueError("Positive measured timings and trial count required")
    median = statistics.median(samples)
    upper = max(samples)
    # Engine writes initially, at every interval, and again at finalization.
    saves = target_steps // checkpoint_interval + 2
    nominal = setup_sec + target_steps * median + saves * checkpoint_sec
    conservative = setup_sec + slowdown * (target_steps * upper + saves * checkpoint_sec)
    return {"median_ms_per_step": median * 1000,
            "trial_steps_per_sec": trials / median,
            "nominal_hours_for_target": nominal / 3600,
            "conservative_hours_for_target": conservative / 3600,
            "nominal_completed_trials_per_hour": trials / (nominal / 3600),
            "checkpoint_count_in_projection": saves,
            "projection_only": True}


def wait_flag(path, timeout):
    until = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= until:
            raise TimeoutError(f"Coordinator did not release {path.name}")
        time.sleep(0.02)


def worker(spec_path):
    spec = json.loads(spec_path.read_text())
    control = Path(spec["control"])
    index = spec["worker"]
    writer = None
    torch.set_num_threads(1)
    try:
        torch.cuda.set_device(0)
        with tempfile.TemporaryDirectory(prefix=f"worker-{index}-", dir=spec["scratch"]) as temporary:
            started = time.perf_counter()
            config = Config(cellspace_path=str(ROOT / "Sample/Cellspace/BCA-IP.yaml"),
                rule_paths=[str(ROOT / "Sample/rule/base-rule.yaml")],
                spatial_event_file_path=str(ROOT / "Sample/Specialevent/BCA-IP_event.py"),
                device="cuda:0", trials=spec["trials"],
                trial_ids=tuple(range(spec["trial_start"], spec["trial_start"] + spec["trials"])),
                execution_mode="cuda", rng_mode="independent", seed=spec["seed"],
                global_prob=spec["global_prob"], quiet=True, use_tqdm="false", log_level="warning",
                stream_dir=temporary, flush_interval=1000, checkpoint_interval=10000,
                candidate_capacity=4096, distributed_mode="off")
            engine = Engine(config)
            sim = engine.state.simulator
            writer = HistoryWriter(config, sim)
            torch.cuda.synchronize()
            atomic_json(control / f"{index}.allocated.json", {
                "setup_sec": time.perf_counter() - started,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "trial_start": spec["trial_start"], "trials": spec["trials"]})
            wait_flag(control / "warmup.go", spec["timeout"])
            for _ in range(spec["warmup"]):
                engine.stepper(sim._current_step)
            torch.cuda.synchronize()
            atomic_json(control / f"{index}.warmed.json", {"step": sim._current_step})
            for repeat in range(spec["repeats"]):
                wait_flag(control / f"repeat-{repeat}.go", spec["timeout"])
                started = time.perf_counter()
                for _ in range(spec["steps"]):
                    engine.stepper(sim._current_step)
                torch.cuda.synchronize()
                atomic_json(control / f"{index}.repeat-{repeat}.json", {
                    "seconds": time.perf_counter() - started,
                    "next_step": sim._current_step,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            wait_flag(control / "checkpoint.go", spec["timeout"])
            started = time.perf_counter()
            writer.checkpoint()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result = {"flush_checkpoint_sec": elapsed,
                      "checkpoint_bytes": (Path(temporary) / "checkpoint.pt").stat().st_size,
                      "current_step": sim._current_step,
                      "history_chunks": len(writer.manifest["chunks"]),
                      "history_records": sum(c["records"] for c in writer.manifest["chunks"]),
                      "identity": writer.manifest["identity"],
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "final_candidate_overflow_rows": int((sim.candidate_plan.cuda.counts > 4096).sum().item())}
            atomic_json(control / f"{index}.checkpoint.json", result)
            wait_flag(control / "finish.go", spec["timeout"])
            writer.close()
            writer = None
    except BaseException as error:
        atomic_json(control / f"{index}.error.json", {
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        raise
    finally:
        if writer is not None:
            writer.close()


class ProbeFailure(RuntimeError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(str(errors))


def measure_cohort(args, out, trials, native_limit, *, label, steps, repeats, warmup):
    partitions = partition_trials(trials, native_limit)
    control = out / label
    control.mkdir()
    scratch = control / "scratch"
    scratch.mkdir()
    processes, logs = [], []
    started = time.perf_counter()
    deadline = time.monotonic() + args.probe_timeout
    peak_used = 0
    memory_samples = []
    last_sample = 0.0

    def sample_memory(force=False):
        nonlocal peak_used, last_sample
        now = time.monotonic()
        if force or now - last_sample >= 0.5:
            free, total = torch.cuda.mem_get_info()
            peak_used = max(peak_used, total - free)
            memory_samples.append([time.perf_counter() - started, total - free])
            last_sample = now

    def wait_all(suffix):
        while True:
            sample_memory()
            errors = [json.loads(p.read_text()) for p in control.glob("*.error.json")]
            if errors:
                raise ProbeFailure(errors)
            if all((control / f"{i}.{suffix}.json").exists() for i in range(len(partitions))):
                sample_memory(force=True)
                return [json.loads((control / f"{i}.{suffix}.json").read_text())
                        for i in range(len(partitions))]
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Benchmark worker exited early: {process.returncode}; {control}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Probe exceeded {args.probe_timeout}s: {control}")
            time.sleep(0.02)

    result = {"trials": trials, "workers_on_same_gpu": len(partitions),
              "partitions": partitions, "steps_per_repeat": steps,
              "repeats": repeats, "warmup_steps": warmup, "label": label}
    print(f"Measuring {label}: {trials} trials, {len(partitions)} process(es), one GPU", flush=True)
    try:
        sample_memory(force=True)
        for i, part in enumerate(partitions):
            spec = {"control": str(control), "scratch": str(scratch), "worker": i,
                    "trials": part["count"], "trial_start": part["start"],
                    "seed": args.seed, "global_prob": args.global_prob,
                    "steps": steps, "repeats": repeats, "warmup": warmup,
                    "timeout": args.probe_timeout}
            spec_path = control / f"{i}.spec.json"
            atomic_json(spec_path, spec)
            log = (control / f"{i}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                "--worker", str(spec_path)], stdout=log, stderr=subprocess.STDOUT))
        allocations = wait_all("allocated")
        setup_sec = time.perf_counter() - started
        atomic_json(control / "warmup.go", {})
        wait_all("warmed")
        seconds_per_step, worker_timings = [], []
        for repeat in range(repeats):
            start = time.perf_counter()
            atomic_json(control / f"repeat-{repeat}.go", {})
            timed = wait_all(f"repeat-{repeat}")
            seconds_per_step.append((time.perf_counter() - start) / steps)
            worker_timings.append(timed)
        start = time.perf_counter()
        atomic_json(control / "checkpoint.go", {})
        checkpoints = wait_all("checkpoint")
        checkpoint_sec = time.perf_counter() - start
        atomic_json(control / "finish.go", {})
        for process in processes:
            if process.wait(timeout=60):
                raise RuntimeError(f"Worker failed during cleanup; see {control}")
        result.update({"status": "ok", "seconds_per_step_samples": seconds_per_step,
            "setup_sec": setup_sec, "cohort_flush_checkpoint_sec": checkpoint_sec,
            "allocations": allocations, "worker_timings": worker_timings,
            "checkpoints": checkpoints,
            "peak_worker_reserved_bytes_sum": sum(r["peak_reserved_bytes"] for r in checkpoints),
            **project_runtime(seconds_per_step, checkpoint_sec, setup_sec, trials=trials,
                              target_steps=args.target_steps, slowdown=args.slowdown_factor)})
    except ProbeFailure as error:
        if not all(e["oom"] for e in error.errors):
            raise
        result.update({"status": "oom", "errors": error.errors})
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for log in logs:
            log.close()
        shutil.rmtree(scratch)
    result.update({"sampled_device_peak_used_bytes": peak_used,
                   "device_memory_samples": memory_samples,
                   "memory_fits": result["status"] == "ok" and peak_used <= args.memory_limit_bytes})
    atomic_json(control / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k in
                     {"label", "trials", "status", "memory_fits", "workers_on_same_gpu",
                      "sampled_device_peak_used_bytes", "trial_steps_per_sec",
                      "nominal_hours_for_target", "conservative_hours_for_target"}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--global-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026092501)
    parser.add_argument("--target-steps", type=int, default=3_000_000)
    parser.add_argument("--memory-fraction", type=float, default=0.96)
    parser.add_argument("--maximum-trials", type=int, default=8192)
    parser.add_argument("--batch-quantum", type=int, default=64)
    parser.add_argument("--profile-steps", type=int, default=1000)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--probe-steps", type=int, default=64)
    parser.add_argument("--probe-timeout", type=int, default=7200)
    parser.add_argument("--slowdown-factor", type=float, default=1.10)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if (not args.output_dir or not 0 < args.memory_fraction < 1 or not 0 <= args.global_prob <= 1
        or args.maximum_trials < 512 or args.batch_quantum < 1 or 512 % args.batch_quantum
        or args.profile_steps < 1000 or args.profile_repeats < 2 or args.probe_steps < 1
        or args.target_steps < 1 or args.probe_timeout < 1 or args.slowdown_factor < 1):
        parser.error("Invalid capacity/profile configuration")
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Request exactly one GPU for this benchmark")
    props = torch.cuda.get_device_properties(0)
    grid = load_cell_space_yaml_to_numpy(str(ROOT / "Sample/Cellspace/BCA-IP.yaml"), include_offset=False)
    rule_count = len(load_transition_rules_yaml(str(ROOT / "Sample/rule/base-rule.yaml")))
    native_limit = min(65535, (2**31 - 1) // grid.size,
                       (2**31 - 1) // (rule_count * min(4096, grid.size)))
    args.memory_limit_bytes = int(props.total_memory * args.memory_fraction)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "Sample/Cellspace/BCA-IP.yaml", ROOT / "Sample/rule/base-rule.yaml",
              ROOT / "Sample/Specialevent/BCA-IP_event.py", Path(__file__)]
    inputs += [p for folder in ["core", "api"] for p in (ROOT / "src/PyBCA" / folder).rglob("*")
               if p.suffix in {".py", ".cu"}]
    plan = {"schema": "pybca-single-gpu-capacity-v1", "completed": False,
            "created_at": datetime.now(timezone.utc).isoformat(), "gpu": props.name,
            "gpu_total_memory_bytes": props.total_memory, "memory_limit_bytes": args.memory_limit_bytes,
            "memory_fraction": args.memory_fraction, "native_process_trial_limit": native_limit,
            "global_prob": args.global_prob, "seed": args.seed, "target_steps": args.target_steps,
            "batch_quantum": args.batch_quantum, "slowdown_factor": args.slowdown_factor,
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(inputs)}, "measurements": [], "production_ready": False,
            "notes": ["Benchmark cohorts are not statistical experimental trials.",
                      "Target runtime is an extrapolation, not a completed 3000000-step run.",
                      "Profiles include event streaming; checkpoints are separately timed on the output filesystem.",
                      "GPU usage includes simultaneous CUDA contexts and is sampled every 0.5 seconds.",
                      "Two workers use disjoint global trial IDs on the same physical GPU, without MPS."]}
    atomic_json(out / "plan.json", plan)
    measured = {}

    def measure(units):
        trials = units * args.batch_quantum
        if trials not in measured:
            record = measure_cohort(args, out, trials, native_limit,
                label=f"capacity-{trials}", steps=args.probe_steps, repeats=1, warmup=10)
            measured[trials] = record
            plan["measurements"].append(record)
            atomic_json(out / "plan.json", plan)
        return measured[trials]["memory_fits"]

    baseline = measure_cohort(args, out, 512, native_limit, label="profile-512",
                             steps=args.profile_steps, repeats=args.profile_repeats, warmup=1000)
    measured[512] = baseline
    plan["measurements"].append(baseline)
    plan["profile_512"] = baseline
    atomic_json(out / "plan.json", plan)
    if baseline["status"] != "ok":
        raise RuntimeError("The requested 512-trial baseline did not complete")
    maximum = min(args.maximum_trials, 2 * native_limit) // args.batch_quantum
    units = largest_measured_batch(measure, initial=512 // args.batch_quantum, maximum=maximum)
    chosen = units * args.batch_quantum
    plan["largest_short_probe_fitting_memory"] = chosen
    atomic_json(out / "plan.json", plan)
    while chosen:
        profile = baseline if chosen == 512 else measure_cohort(args, out, chosen, native_limit,
            label=f"profile-memory-{chosen}", steps=args.profile_steps,
            repeats=args.profile_repeats, warmup=100)
        if chosen != 512:
            plan["measurements"].append(profile)
            atomic_json(out / "plan.json", plan)
        if profile["memory_fits"]:
            plan["profile_near_full_memory"] = profile
            break
        chosen -= args.batch_quantum
    plan["recommended_memory_cohort"] = chosen
    plan["completed"] = True
    atomic_json(out / "plan.json", plan)
    if not chosen:
        raise RuntimeError("No measured cohort fit the memory budget")


if __name__ == "__main__":
    main()
