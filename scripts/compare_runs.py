"""Compare streamed trajectories' final states and complete count histories.

Works with one GPU and torchrun directories and arbitrary global trial IDs.
The digest is order-independent across chunks and ranks, but includes every
event/rule occurrence with its step and multiplicity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from PyBCA.api.streaming import iter_history


def summary(directory):
    root = Path(directory)
    manifests = ([root/"manifest.json"] if (root/"manifest.json").exists()
                 else sorted(root.glob("rank_*/manifest.json")))
    if not manifests:
        raise ValueError(f"No run found: {directory}")
    states = {}
    # 256-bit additive multiset digest avoids accumulating millions of rows.
    digest, count = 0, 0
    names = Counter()
    steps = set()
    for manifest in manifests:
        metadata = json.loads(manifest.read_text())
        state = torch.load(manifest.parent/"checkpoint.pt", map_location="cpu", weights_only=True)
        if state["manifest"]["identity"] != metadata["identity"]:
            raise ValueError("Checkpoint and history manifest describe different runs")
        steps.add(state["next_step"])
        if state["next_step"] != metadata["next_step"]:
            raise ValueError("Run must be checkpointed at the end of its history before comparison")
        for i, trial in enumerate(metadata["trial_ids"]):
            if str(trial) in states:
                raise ValueError("Duplicate trial ID across ranks")
            layout = {"shape": list(state["cells"][i].shape), "offset": state["offset"],
                      "dtype": str(state["cells"].dtype), "step": state["next_step"]}
            states[str(trial)] = hashlib.sha256(json.dumps(layout).encode()+state["cells"][i].numpy().tobytes()).hexdigest()
        for row in iter_history(manifest.parent, verify=True):
            key = [row["kind"], row["trial"], row["step"], row["name"]]
            token = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest(), "little")
            digest = (digest + token*row["count"]) % 2**256
            count += row["count"]
            names[row["name"]] += row["count"]
    return {"steps": sorted(steps), "states": states, "history_digest": f"{digest:064x}",
            "history_occurrences": count, "counts_by_name": dict(sorted(names.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second", nargs="?")
    parser.add_argument("--output")
    args = parser.parse_args()
    first = summary(args.first)
    result = {"first": first}
    if args.second:
        second = summary(args.second)
        result.update(second=second, equal=first == second)
    text = json.dumps(result, indent=2)+"\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    if args.second and not result["equal"]:
        raise SystemExit("Runs differ")


if __name__ == "__main__":
    main()
