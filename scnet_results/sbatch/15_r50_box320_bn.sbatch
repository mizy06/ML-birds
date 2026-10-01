#!/bin/bash
#SBATCH --job-name=cub_r50_b320bn
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:1
#SBATCH --mem=56G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment=workbuddy-cub-15-r50-box320-bn
# ---------------------------------------------------------------------------
# bn-train 2x2, cell (320, bbox): r50_box + --bn-train.
# ---------------------------------------------------------------------------

module purge
module load compiler/dtk/25.04

ROOT="$HOME/cub_job"
PY="$ROOT/venv/bin/python"
export CUB_DATA_ROOT="$ROOT/data"
export CUB_OUT_ROOT="$ROOT/runs"
export TORCH_HOME="$HOME/.cache/torch"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$ROOT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} CPUS=${SLURM_CPUS_PER_TASK} GRES=${SLURM_JOB_GRES:-dcu:1}"
echo "START=$(date '+%F %T')"
"$PY" -u cub_experiments_scnet.py --device-info

"$PY" -u cub_experiments_scnet.py \
    --data-root "$CUB_DATA_ROOT" \
    --out-root "$CUB_OUT_ROOT" \
    --split-file train_test_split_0to1_1to9.txt \
    --model resnet50 --size 320 --bbox --run r50_box320_bn --bn-train \
    --workers 8
rc=$?
echo "EXIT_CODE=$rc"
echo "END=$(date '+%F %T')"
exit $rc
