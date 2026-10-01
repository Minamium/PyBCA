"""Measure CUDA event spans for existing BCA-IP stages without editing kernels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile

import torch

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import HistoryWriter

ROOT = Path(__file__).resolve().parents[1]
STAGES = {0: "match_and_pack", 2: "rule_order_conflict_and_write", 3: "events"}


def profile(trials, steps):
    with tempfile.TemporaryDirectory(prefix="pybca-stage-profile-") as folder:
        config = Config(cellspace_path=str(ROOT / "Sample/Cellspace/BCA-IP.yaml"),
            rule_paths=[str(ROOT / "Sample/rule/base-rule.yaml")],
            spatial_event_file_path=str(ROOT / "Sample/Specialevent/BCA-IP_event.py"),
            trials=trials, device="cuda", execution_mode="cuda", rng_mode="independent",
            global_prob=0.5, seed=2026092501, quiet=True, use_tqdm="false",
            log_level="warning", stream_dir=folder, flush_interval=1000)
        engine = Engine(config)
        sim = engine.state.simulator
        writer = HistoryWriter(config, sim)
        plan = sim.candidate_plan.cuda
        launch = plan.launch
        spans = {name: [] for name in STAGES.values()}

        def timed_launch(op, *args, **kwargs):
            before = torch.cuda.Event(enable_timing=True)
            after = torch.cuda.Event(enable_timing=True)
            before.record()
            result = launch(op, *args, **kwargs)
            after.record()
            spans[STAGES[op]].append((before, after))
            return result

        try:
            for _ in range(10):
                engine.stepper(sim._current_step)
            torch.cuda.synchronize()
            plan.launch = timed_launch
            for _ in range(steps):
                engine.stepper(sim._current_step)
            torch.cuda.synchronize()
            result = {"trials": trials, "steps": steps, "warmup": 10,
                      "global_prob": 0.5, "seed": 2026092501,
                      "stage_ms_samples": {name: [a.elapsed_time(b) for a, b in pairs]
                                           for name, pairs in spans.items()}}
            result["stage_ms_medians"] = {name: statistics.median(values)
                                          for name, values in result["stage_ms_samples"].items()}
            return result
        finally:
            plan.launch = launch
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "method": "CUDA event spans; compound stages include inter-kernel gaps. Instrumented diagnostic, not the primary throughput measurement.",
        "measurements": [profile(n, 50) for n in [64, 512]]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage_profiles": [r["stage_ms_medians"] for r in result["measurements"]]}))


if __name__ == "__main__":
    main()
