"""Freeze a committed history prefix without stopping or modifying its writer.

The snapshot contains analysis manifests and read-only references to immutable
history chunks, never restart checkpoints. The existing decoder verifies every
referenced chunk before extracting events. No classification policy is changed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import check_bca_ip_variants as checks
from continue_bca_ip_variants import rank_config
from PyBCA.api.streaming import identity, sha256

ROOT = Path(__file__).resolve().parents[1]


def history_prefix(manifest, horizon):
    if horizon <= 0 or horizon > manifest["next_step"]:
        raise ValueError("Requested horizon is not committed")
    result = copy.deepcopy(manifest)
    chunks = []
    previous = 0
    for chunk in manifest["chunks"]:
        if chunk["next_step"] > horizon:
            break
        path = Path(chunk["path"])
        if path.is_absolute() or path.parts[0] != "history" or ".." in path.parts:
            raise ValueError("Unexpected history path")
        if chunk["start_step"] != previous or chunk["next_step"] <= previous:
            raise ValueError("History has a gap or overlap")
        chunks.append(copy.deepcopy(chunk))
        previous = chunk["next_step"]
    if previous != horizon:
        raise ValueError("Horizon must be an exact committed chunk boundary")
    result["next_step"], result["chunks"] = horizon, chunks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--variant", required=True, choices=checks.VARIANTS)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("Snapshot must be outside the production run")
    if args.horizon % 10000 or args.horizon < 100000:
        raise ValueError("Use the decoder's 10k-bin endpoint")
    checks.inputs()
    # Read each atomically replaced manifest once, before long extraction begins.
    captures = [(rank, (run / f"rank_{rank:04d}/manifest.json").read_bytes())
                for rank in range(8)]
    output.mkdir(parents=True, exist_ok=False)
    frozen_root = output / "snapshot"
    ranks = []
    for rank, raw in captures:
        original = json.loads(raw)
        if original["identity"] != identity(rank_config(args.variant, rank)):
            raise ValueError("Input, dynamics, RNG or trial identity changed")
        if original["trial_ids"] != list(range(rank * 64, (rank + 1) * 64)):
            raise ValueError("Incorrect trial roster")
        frozen = history_prefix(original, args.horizon)
        directory = frozen_root / f"rank_{rank:04d}"
        directory.mkdir(parents=True)
        (directory / "source-manifest.json").write_bytes(raw)
        path = directory / "manifest.json"
        path.write_text(json.dumps(frozen, indent=2) + "\n")
        (directory / "history").symlink_to(run / directory.name / "history", target_is_directory=True)
        ranks.append({"rank": rank, "observed_step": original["next_step"],
                      "analysis_step": args.horizon, "chunks": len(frozen["chunks"]),
                      "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                      "frozen_manifest_sha256": sha256(path),
                      "identity": original["identity"]})
    report = {"captured_at": datetime.now(timezone.utc).isoformat(), "passed": False,
              "variant": args.variant, "run": str(run), "horizon": args.horizon,
              "trials": 512, "ranks": ranks, "source": checks.source_stamp(),
              "snapshot_script_sha256": sha256(__file__),
              "scope": "Analysis-only committed history prefix; not a restartable checkpoint."}
    report_path = output / "snapshot.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    instance = checks.MAPS / f"condition{args.variant[9]}.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/read_bca_ip_fsm_outputs.py"),
                    str(frozen_root), "--instance", str(instance),
                    "--output", str(output / "readout"), "--horizon", str(args.horizon)], check=True)
    meta = json.loads((output / "readout/summary.json").read_text())
    if meta["provenance"]["checked_chunks"] != sum(r["chunks"] for r in ranks):
        raise ValueError("Not all frozen chunks were verified")
    report["passed"] = True
    report["extraction"] = meta["provenance"]
    report["files"] = {str(p.relative_to(output)): sha256(p)
                       for p in sorted((output / "readout").iterdir()) if p.is_file()}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"snapshot_passed": True, "variant": args.variant,
                      "horizon": args.horizon, "files": report["files"]}), flush=True)


if __name__ == "__main__":
    main()
