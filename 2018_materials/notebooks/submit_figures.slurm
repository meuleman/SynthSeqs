#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=f.o
#SBATCH --error=f.e
#SBATCH --gres=gpu:p100
#SBATCH --partition=gpu1
#SBATCH --job-name=figs
#
# ==== Resource Requirements and Limits
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=1
#
# ==== Standard Environment
export TMPDIR=/tmp/slurm.${SLURM_JOBID}
# ==== end of preamble

module load openssl-dev/1.0.1t
module load cuda/8.0.44
module load libsodium/1.0.9
module load sqlite/3.18.2
module load cudnn/5.1
module load tensorflow-gpu/20170719
module load zeromq/4.1.4 

python -u classifier_figures.py 
