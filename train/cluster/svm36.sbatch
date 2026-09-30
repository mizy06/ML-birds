#!/bin/bash
#SBATCH --job-name=svm36
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=dcu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --comment=workbuddy-svm36

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

echo "HOST=$(hostname) JOB=$SLURM_JOB_ID CPUS=$SLURM_CPUS_PER_TASK"
echo "python: $(which python3)"
python3 -u "$SLURM_SUBMIT_DIR/train_svm.py"
echo "EXIT_CODE=$?"
