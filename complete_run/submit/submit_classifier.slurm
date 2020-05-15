#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=submit/out/classifier.o
#SBATCH --error=submit/out/classifier.e
#SBATCH --gres=gpu:p100
#SBATCH --partition=gpu1
#SBATCH --job-name=classifier
#
# ==== Resource Requirements and Limits
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
#
# ==== Standard Environment
export TMPDIR=/tmp/slurm.${SLURM_JOBID}
# ==== end of preamble

python3 -m classifier
