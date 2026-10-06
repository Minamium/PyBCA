"""Submit the validated three-condition experiment, using at most eight GPUs.

Run from the deployed repository root on rokko. Each accepted job ID is saved
before submitting another job. An existing record prevents duplicate submits.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    deployment = json.loads((root / "deployment.json").read_text())
    path = root / "results/variants-20261007/submission.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"submitted_at": datetime.now(timezone.utc).isoformat(),
              "deployment": deployment, "workspace": str(root), "jobs": []}
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)

    def submit(label, script, *options):
        command = ["qsub", *options, script]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+\.[A-Za-z0-9_.-]+", job_id):
            raise RuntimeError(f"Unexpected qsub reply; inspect qstat before retrying: {result.stdout!r}")
        row = {"label": label, "job_id": job_id, "command": command,
               "qsub_stderr": result.stderr.strip()}
        record["jobs"].append(row)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        print(json.dumps(row), flush=True)
        return job_id

    preflight = submit("preflight", "scripts/rokko/bca-ip-variants-preflight.pbs")
    previous = None
    for variant, name in (("condition1_N1", "bca_c1n1_3m"),
                          ("condition1_N2", "bca_c1n2_3m"),
                          ("condition2_N2", "bca_c2n2_3m")):
        dependency = f"afterok:{preflight}"
        if previous:
            dependency += f",afterany:{previous}"
        previous = submit(variant, "scripts/rokko/bca-ip-variant-3m.pbs",
                          "-N", name, "-v", f"VARIANT={variant}",
                          "-W", f"depend={dependency}")


if __name__ == "__main__":
    main()
