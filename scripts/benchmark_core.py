"""Measure BCA-IP wall time and allocated CUDA memory, including event handling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import HistoryWriter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "torch_sparse", "cuda"], required=True)
    parser.add_argument("--rng", choices=["legacy", "independent"], default="legacy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    with tempfile.TemporaryDirectory(prefix="pybca-bench-") as td:
        c = Config(cellspace_path=str(ROOT/"Sample/Cellspace/BCA-IP.yaml"),
                   rule_paths=[str(ROOT/"Sample/rule/base-rule.yaml")],
                   spatial_event_file_path=str(ROOT/"Sample/Specialevent/BCA-IP_event.py"),
                   device=args.device, trials=args.trials, execution_mode=args.mode,
                   rng_mode=args.rng, seed=31, quiet=True, use_tqdm="false", log_level="warning",
                   stream_dir=td, flush_interval=1000, checkpoint_interval=10000)
        started = time.perf_counter()
        engine = Engine(c)
        sim = engine.state.simulator
        writer = HistoryWriter(c, sim)
        setup = time.perf_counter()-started
        try:
            for step in range(10):
                engine.stepper(step)
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            samples = []
            for _ in range(args.repeats):
                sync()
                started = time.perf_counter()
                for step in range(args.steps):
                    engine.stepper(step)
                sync()
                samples.append((time.perf_counter()-started)/args.steps)
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            started = time.perf_counter()
            writer.checkpoint()
            save = time.perf_counter()-started
            overflow_rows = (int((sim.candidate_plan.cuda.counts > sim.candidate_capacity).sum().item())
                             if args.mode == "cuda" else None)
            result = {"mode": args.mode, "rng": args.rng, "trials": args.trials,
                      "steps_per_repeat": args.steps, "seconds_per_step_samples": samples,
                      "median_ms_per_step": statistics.median(samples)*1000,
                      "trial_steps_per_sec": args.trials/statistics.median(samples),
                      "peak_allocated_bytes": peak, "peak_reserved_bytes": reserved,
                      "setup_sec": setup, "flush_checkpoint_sec": save,
                      "final_candidate_overflow_rows": overflow_rows,
                      "torch": str(torch.__version__), "device": args.device,
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      "shape": list(sim.TCHW.shape), "rules": len(sim.rule_ids)}
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps(result), flush=True)
        finally:
            writer.close()


if __name__ == "__main__":
    main()
