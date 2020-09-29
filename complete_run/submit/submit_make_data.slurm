#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=submit/out/make_data.o
#SBATCH --error=submit/out/make_data.e
#SBATCH --partition=queue0
#SBATCH --job-name=make_data
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

python3 -u -m make_data 
