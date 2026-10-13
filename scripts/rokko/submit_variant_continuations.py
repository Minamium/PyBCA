"""Queue one restart preflight and three serial 8-GPU continuation jobs.

Each qsub intent and reply is persisted. An existing record requires manual
inspection rather than risking duplicate submissions after a network failure.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from continue_bca_ip_variants import OUTPUT_ROOT, prepared
from PyBCA.api.streaming import atomic_json, sha256


def main():
    os.chdir(ROOT)
    prepared(OUTPUT_ROOT)
    path = OUTPUT_ROOT / "submission.json"
    record = {"submitted_at": datetime.now(timezone.utc).isoformat(),
              "workspace": str(ROOT), "jobs": [], "pending": None,
              "continuation_sha256": sha256(OUTPUT_ROOT / "continuation.json"),
              "maximum_simultaneous_gpus": 8}
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    def submit(label, script, *options):
        command = ["qsub", *options, script]
        record["pending"] = {"label": label, "command": command}
        atomic_json(path, record)
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", job_id):
            raise RuntimeError(f"Unexpected qsub reply; inspect qstat before retrying: {result.stdout!r}")
        row = {"label": label, "job_id": job_id, "command": command,
               "qsub_stderr": result.stderr.strip()}
        record["jobs"].append(row)
        record["pending"] = None
        atomic_json(path, record)
        print(json.dumps(row), flush=True)
        return job_id

    preflight = submit("preflight", "scripts/rokko/bca-ip-continue-preflight.pbs")
    previous = None
    for variant, name in (("condition1_N1", "bca_c1n1_6m"),
                          ("condition1_N2", "bca_c1n2_6m"),
                          ("condition2_N2", "bca_c2n2_6m")):
        dependency = f"afterok:{preflight}"
        if previous:
            dependency += f",afterany:{previous}"
        previous = submit(variant, "scripts/rokko/bca-ip-variant-continue-6m.pbs",
                          "-N", name, "-v", f"VARIANT={variant}",
                          "-W", f"depend={dependency}")


if __name__ == "__main__":
    main()
