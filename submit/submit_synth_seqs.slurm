#!/bin/bash
#
# based on Slurm workload manager
#
# ==== Job Attributes
#SBATCH --output=submit/out/synth_seqs.o
#SBATCH --error=submit/out/synth_seqs.e
#SBATCH --gres=gpu:p100
#SBATCH --partition=gpu1
#SBATCH --job-name=synth_seqs
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

# Set up output directories
python3 setup_dirs.py -n mpra 

# Generate and optimize sequences
python3 -u -m synth_seqs -n 1000 -c $SLURM_ARRAY_TASK_ID --seed 42 -i 10000 --save_interval 10 --run_name mpra
