#!/bin/bash

#SBATCH --job-name=flash-attn
#SBATCH --output=flash-attn-%j.out
#SBATCH --error=flash-attn-%j.err
#SBATCH --partition=normal
#SBATCH --account=ch17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells.
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT:-29500}"

srun --export=ALL python flash_attn_experiment_geo_parallelization.py
