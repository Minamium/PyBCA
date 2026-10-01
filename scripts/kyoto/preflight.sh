#!/bin/bash
#SBATCH -p pg
#SBATCH -t 02:00:00
#SBATCH --rsc g=1
#SBATCH -J pybca_a100_check
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
set -euo pipefail
module purge
module load SysG/2022 PrgEnvNvidia/2023 cuda/12.1.1 pytorch/2.2.0.py311_cuda-12.1
cd "${SLURM_SUBMIT_DIR}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export PATH=/home/b/b39859/PyBCA_workspace/venv/bin:$PATH
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYBCA_TEST_DEVICE=cuda PYBCA_CUDA_CACHE="$PWD/results/cuda-cache"
out="$PWD/results/preflight-${SLURM_JOB_ID}"
mkdir -p "$out"
exec > >(tee "$out/console.log") 2>&1
python - <<'PY'
import json, os, platform, subprocess
import numpy as np
import torch
from PyBCA.api.streaming import atomic_json
assert torch.cuda.device_count() == 1, "This preflight requests exactly one GPU"
p = torch.cuda.get_device_properties(0)
assert "A100" in p.name and p.total_memory > 75 * 1024**3, (p.name, p.total_memory)
assert torch.from_numpy(np.zeros(2, dtype=np.int8)).numpy().dtype == np.int8
report = {"python": platform.python_version(), "torch": str(torch.__version__),
          "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": p.name,
          "memory_bytes": p.total_memory, "capability": list(torch.cuda.get_device_capability(0)),
          "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)}
atomic_json(f"results/preflight-{os.environ['SLURM_JOB_ID']}/environment.json", report)
print(json.dumps(report), flush=True)
PY
case "${PYBCA_PREFLIGHT_PHASE:-all}" in
  all) python tests/test_rule_equivalence.py --device cuda --output "$out/all-rules.json" ;;
  replay) ;; # Resume after a separately recorded successful full rule audit.
  *) exit 2 ;;
esac
python -m unittest discover -s tests -p 'test_core_runtime.py' -v
python scripts/run_bca_ip.py --output-dir "$out/replay" --trials 2 \
  --steps 100000 --seed 20260924 --global-prob 0.5 --trial-start 0 \
  --device cuda --mode cuda --rng independent --flush-interval 1000 \
  --checkpoint-interval 10000 --candidate-capacity 4096 --label a100-v100-replay
python scripts/validate_device_replay.py \
  --reference-checkpoint reference/v100-p05-checkpoint.pt \
  --reference-events docs/experiments/2026-09-25-probability-check/comparison.json \
  --replay "$out/replay" --output "$out/device-parity.json"
