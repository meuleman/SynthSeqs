#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=submit/out/optimizer.o
#SBATCH --error=submit/out/optimizer.e
#SBATCH --gres=gpu:p100
#SBATCH --partition=gpu1
#SBATCH --job-name=optimize
#
# ==== Resource Requirements and Limits
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=1
#
# ==== Standard Environment
export TMPDIR=/tmp/slurm.${SLURM_JOBID}
# ==== end of preamble

python3 -u -m optimize -n 10 -c 9 --seed 42 -i 1000 --save_interval 50 --run_name "test" 
