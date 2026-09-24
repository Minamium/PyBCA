#!/bin/bash
#SBATCH -p pg
#SBATCH -t 01:00:00
#SBATCH --rsc g=1
#SBATCH -J pybca_a100_full
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
out="$PWD/results/a100-capacity-profile-${SLURM_JOB_ID}"
mkdir -p results/logs
exec > >(tee "results/logs/a100-capacity-profile-${SLURM_JOB_ID}.log") 2>&1
nvidia-smi
python scripts/profile_bca_ip_capacity.py \
  --source-plan "${PYBCA_SOURCE_PLAN:?Set source capacity plan}" \
  --output-dir "$out" --seconds-per-repeat 200 --repeats 3
python scripts/profile_cuda_stages.py --output "$out/cuda-stages.json"
nvidia-smi
