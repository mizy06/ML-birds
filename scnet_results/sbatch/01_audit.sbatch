#!/bin/bash
#SBATCH --job-name=cub_audit
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=dcu:1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --comment=workbuddy-cub-01-audit
# ---------------------------------------------------------------------------
# Example 1 of cub_experiments.py:
#   python cub_experiments.py --data-root D:\A\ML\data --audit
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
    --audit
rc=$?
echo "EXIT_CODE=$rc"
echo "END=$(date '+%F %T')"
exit $rc
