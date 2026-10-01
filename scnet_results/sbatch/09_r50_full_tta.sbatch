#!/bin/bash
#SBATCH --job-name=cub_r50_tta
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment=workbuddy-cub-09-r50-tta
# ---------------------------------------------------------------------------
# Example 6 of cub_experiments.py, applied to r50_full:
#   python cub_experiments.py --model resnet50 --size 224 --eval-only \
#       --checkpoint runs/r50_full/best.pt --subset val --tta
#
# --split-file is passed explicitly: in --eval-only mode the script verifies the
# checkpoint's recorded split_hash against the current split and refuses to run
# on a mismatch. This is the user's custom split, not the official CUB one.
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
    --model resnet50 --size 224 \
    --eval-only --checkpoint "$CUB_OUT_ROOT/r50_full/best.pt" --subset val --tta \
    --workers 4
rc=$?
echo "EXIT_CODE=$rc"
echo "END=$(date '+%F %T')"
exit $rc
