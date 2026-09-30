#!/bin/bash
#SBATCH --job-name=svm36x
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=dcu:1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment=workbuddy-svm36x

module purge
module load anaconda3/2023.09

cd "$SLURM_SUBMIT_DIR"

export ML_DATA_DIR="$SLURM_SUBMIT_DIR/data"
export N_JOBS=36
export JOBLIB_TEMP_FOLDER="$SLURM_SUBMIT_DIR/.joblib_tmp"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$JOBLIB_TEMP_FOLDER"

echo "HOST=$(hostname) JOB=$SLURM_JOB_ID N_JOBS=$N_JOBS"
echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE nproc=$(nproc) alloc_cpus=$SLURM_JOB_CPUS_PER_NODE"
echo "python: $(which python3)"
python3 -u "$SLURM_SUBMIT_DIR/train_svm.py"
echo "EXIT_CODE=$?"
