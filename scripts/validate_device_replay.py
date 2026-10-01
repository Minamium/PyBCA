"""Compare a completed independent-RNG replay with an archived device run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from PyBCA.api.streaming import atomic_json, iter_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-events", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    expected = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=True)
    actual = torch.load(args.replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    expected_events = json.loads(args.reference_events.read_text())["global_prob_0_5"]["rows"]
    actual_events = list(iter_history(args.replay, verify=True))
    canonical = lambda rows: sorted(rows, key=lambda r: (r["trial"], r["step"], r["name"], r["kind"]))
    checks = {
        "identity": expected["manifest"]["identity"] == actual["manifest"]["identity"],
        "steps": expected["next_step"] == actual["next_step"],
        "offset": expected["offset"] == actual["offset"],
        "shape_dtype": expected["cells"].shape == actual["cells"].shape and expected["cells"].dtype == actual["cells"].dtype,
        "all_cells": torch.equal(expected["cells"], actual["cells"]),
        "rule_probabilities": torch.equal(expected["rule_probs"], actual["rule_probs"]),
        "every_event": canonical(expected_events) == canonical(actual_events),
        "final_manifest": actual["manifest"] == json.loads((args.replay / "manifest.json").read_text()),
    }
    result = {"passed": all(checks.values()), "checks": checks,
              "updates": actual["next_step"], "trials": actual["cells"].shape[0],
              "events": len(actual_events), "reference_environment": expected["manifest"]["environment"],
              "replay_environment": actual["manifest"]["environment"],
              "reference_checkpoint_sha256": hashlib.sha256(args.reference_checkpoint.read_bytes()).hexdigest(),
              "reference_cells_sha256": hashlib.sha256(expected["cells"].numpy().tobytes()).hexdigest(),
              "replay_cells_sha256": hashlib.sha256(actual["cells"].numpy().tobytes()).hexdigest()}
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("Cross-device replay differs from the reference")


if __name__ == "__main__":
    main()
