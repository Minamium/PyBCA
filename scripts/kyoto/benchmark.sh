#!/bin/bash
#SBATCH -p pg
#SBATCH -t 06:00:00
#SBATCH --rsc g=1
#SBATCH -J pybca_a100_bench
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
mkdir -p results/logs
exec > >(tee "results/logs/a100-benchmark-${SLURM_JOB_ID}.log") 2>&1
python scripts/benchmark_bca_ip_gpu.py \
  --output-dir "$PWD/results/a100-benchmark-${SLURM_JOB_ID}" \
  --global-prob 0.5 --seed 2026092501 --target-steps 3000000 \
  --memory-fraction 0.96 --maximum-trials 8192 --batch-quantum 64 \
  --profile-steps 1000 --profile-repeats 3 --probe-steps 64
