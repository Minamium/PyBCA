#!/bin/bash
# Source from PBS jobs after changing to their immutable deployment directory.
module load cuda/11.8.0 gcc/11.4.0
export PATH=/home/IM25D029/PyBCA_workspcace/venv/bin:$PATH
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYBCA_CUDA_CACHE="$PWD/results/cuda-cache"
export VARIANT_ROOT="$PWD/results/variants-20261007"
export VARIANT_MAPS="$PWD/Sample/Cellspace/BCA-IP-variants"
mkdir -p results/logs
python - <<'PY'
import json
import torch
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"Expected eight allocated GPUs, found {torch.cuda.device_count()}")
print(json.dumps({"gpu_count": 8, "gpus": [torch.cuda.get_device_name(i) for i in range(8)]}), flush=True)
PY
(cd "$VARIANT_MAPS" && sha256sum --strict -c SHA256SUMS)

run_variant() {
    local variant="$1" output="$2" steps="$3" seed="$4"
    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        scripts/run_bca_ip.py \
        --output-dir "$output" --steps "$steps" --trials 512 \
        --seed "$seed" --global-prob 0.5 --trial-start 0 \
        --device cuda --mode cuda --rng independent \
        --flush-interval 1000 --checkpoint-interval 10000 --candidate-capacity 4096 \
        --cellspace "$VARIANT_MAPS/BCA-IP_${variant}.yaml" \
        --events "$VARIANT_MAPS/BCA-IP_wide_events.py" --label "$variant"
}
