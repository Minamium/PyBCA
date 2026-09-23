"""Reproducible BCA-IP runs with bounded history and checkpoint restart.

Use torchrun for multiple GPUs. `--steps` is the absolute target on resume.
The bundled integrated circuit is labelled legacy-reference: its association
with a particular paper instance is not inferred from the filename.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal

from PyBCA.api import Config, Engine
from PyBCA.api.streaming import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["reference", "torch_sparse", "cuda"], default="cuda")
    parser.add_argument("--rng", choices=["legacy", "independent"], default="independent")
    parser.add_argument("--global-prob", type=float, default=1.)
    parser.add_argument("--flush-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--candidate-capacity", type=int, default=4096)
    parser.add_argument("--record-rule-history", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cellspace", default=str(ROOT/"Sample/Cellspace/BCA-IP.yaml"))
    parser.add_argument("--rules", nargs="+", default=[str(ROOT/"Sample/rule/base-rule.yaml")])
    parser.add_argument("--events", default=str(ROOT/"Sample/Specialevent/BCA-IP_event.py"))
    parser.add_argument("--label", default="legacy-reference")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    distributed = "RANK" in os.environ
    resume = str(out if distributed else out/"checkpoint.pt") if args.resume else None
    config = Config(cellspace_path=args.cellspace, rule_paths=args.rules,
                    spatial_event_file_path=args.events, device=args.device,
                    execution_mode=args.mode, rng_mode=args.rng, trials=args.trials,
                    trial_ids=tuple(range(args.trial_start, args.trial_start+args.trials)),
                    steps=args.steps, seed=args.seed, global_prob=args.global_prob,
                    quiet=True, use_tqdm="false", stream_dir=str(out),
                    flush_interval=args.flush_interval, checkpoint_interval=args.checkpoint_interval,
                    resume_from=resume, candidate_capacity=args.candidate_capacity,
                    record_rule_history=args.record_rule_history,
                    distributed_mode="auto", distributed_run_dir=str(out))
    engine = Engine(config)
    # PBS sends TERM before killing a timed-out job. Finish the current CA step,
    # checkpoint it, and leave a precise status. SIGKILL still falls back to the
    # last periodic checkpoint and the committed history manifest.
    def stop(signum, frame):
        engine.stop_requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = engine.run()
    rank = engine.distributed.context.rank
    summary = {"label": args.label, "rank": rank, "current_step": result.current_step,
               "target_step": args.steps, "elapsed_sec": result.elapsed_sec,
               "stopped": bool(getattr(engine, "stop_requested", False)),
               "history": result.event_history,
               "active": engine.distributed.active,
               "trial_ids": list(engine.config.trial_ids or []) if engine.distributed.active else [],
               "cellspace": str(Path(args.cellspace).resolve()),
               "instance_certified": False}
    target = out/f"rank_{rank:04d}" if distributed else out
    atomic_json(target/"summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
