#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=submit/out/generator.o
#SBATCH --error=submit/out/generator.e
#SBATCH --gres=gpu:p100
#SBATCH --partition=gpu2
#SBATCH --job-name=gan
#
# ==== Resource Requirements and Limits
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
#
# ==== Standard Environment
export TMPDIR=/tmp/slurm.${SLURM_JOBID}
# ==== end of preamble

python3 -u -m generator
