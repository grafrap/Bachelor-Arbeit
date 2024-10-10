#!/bin/bash

#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:20:00

module load stack/.2024-04-silent stack/2024-04 gcc julia openmpi
export JULIA_DEPOT_PATH="$HOME/.julia:/cluster/work/math/hpclab/julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia E-Szi-S2_pll.jl 1