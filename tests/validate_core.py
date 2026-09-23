"""Stepwise differential validation against the reference and legacy engines."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import time

import torch

from simulator_parity import _cases, _resolve
from PyBCA._legacy.cli_simClass import BCA_Simulator as Legacy
from PyBCA.core.simulator import BCA_Simulator


def compare(a, b, label, history=True):
    for attr in ("TCHW", "TNHW_boolMask", "TCHW_applied"):
        if not torch.equal(getattr(a, attr).cpu(), getattr(b, attr).cpu()):
            raise AssertionError(f"{label}: {attr}")
    if a.event_history != b.event_history:
        raise AssertionError(f"{label}: events")
    if history and a.rule_history != b.rule_history:
        raise AssertionError(f"{label}: rules")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bca-steps", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"device": args.device, "torch": str(torch.__version__), "cases": []}
    if args.device.startswith("cuda"):
        report["gpu"] = torch.cuda.get_device_name(0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in _cases():
        started = time.perf_counter()
        kwargs = dict(cellspace_path=_resolve(case.cellspace_path),
                      rule_paths=[_resolve(p) for p in case.rule_paths], device=args.device,
                      spatial_event_filePath=_resolve(case.event_path), use_tqdm=True,
                      trial_constant_sweep=copy.deepcopy(case.trial_constant_sweep))
        with contextlib.redirect_stdout(io.StringIO()):
            baseline = BCA_Simulator(**kwargs, quiet=True, record_rule_history=True)
            baseline.Allocate_torch_Tensors_on_Device()
            baseline.set_ParallelTrial(case.trials)
            reference = copy.deepcopy(baseline)
            from PyBCA.core.optimized import CandidatePlan
            candidate = copy.deepcopy(baseline)
            mode = "cuda" if args.device.startswith("cuda") else "torch_sparse"
            candidate.candidate_plan = CandidatePlan(candidate, mode, "legacy")
            # The original implementation is also checked directly, independently
            # of the reference selector in the refactored simulator.
            legacy = Legacy(**kwargs)
            legacy.Allocate_torch_Tensors_on_Device()
            legacy.set_ParallelTrial(case.trials)
        steps = args.bca_steps if case.name == "bca_ip_large_grid" else case.steps
        for step in range(steps):
            kw = dict(global_prob=case.global_prob, seed=case.seed,
                      state_gate_enable=case.state_gate_enable, state_gate_interval=case.state_gate_interval)
            reference.step(**kw)
            candidate.step(**kw)
            legacy.step(**kw)
            compare(reference, candidate, f"{case.name}/{step}")
            compare(reference, legacy, f"legacy/{case.name}/{step}", history=False)
        result = {"name": case.name, "steps": steps, "trials": case.trials,
                  "matched": ["cells", "accepted_centers", "written_cells", "events", "rule_counts"],
                  "event_count": sum(len(v) for h in (reference.event_history or []) for v in h.values()),
                  "elapsed_sec": time.perf_counter()-started}
        report["cases"].append(result)
        output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(result), flush=True)
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
