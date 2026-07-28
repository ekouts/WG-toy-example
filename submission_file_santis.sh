#!/bin/bash

#SBATCH --job-name=flash-attn
#SBATCH --output=flash-attn-%j.out
#SBATCH --error=flash-attn-%j.err
#SBATCH --partition=normal
#SBATCH --account=ch17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:05:00

uenv run prgenv-gnu/25.6:v2 --view=modules,default -- \
    srun --export=ALL --ntasks=1 --gpus-per-task=4 \
    source .venv/bin/activate /
    python flash_attn_experiment.py