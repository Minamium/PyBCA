#!/bin/bash
#SBATCH -p pg
#SBATCH -t 01:00:00
#SBATCH --rsc g=1
#SBATCH -J pybca_a100_ctrl
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
export PYBCA_CUDA_CACHE="$PWD/results/cuda-cache"
out="$PWD/results/a100-control-${SLURM_JOB_ID}"
mkdir -p "$out"
exec > >(tee "$out/console.log") 2>&1
nvidia-smi
python scripts/benchmark_core.py --mode cuda --rng independent --device cuda \
  --trials 64 --steps 1000 --repeats 3 --global-prob 0.5 --seed 2026092501 \
  --output "$out/trials-64.json"
python scripts/benchmark_core.py --mode cuda --rng independent --device cuda \
  --trials 512 --steps 200 --repeats 3 --global-prob 0.5 --seed 2026092501 \
  --output "$out/trials-512.json"
nvidia-smi
