#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --output=job_seq_Sz1.out
#SBATCH --error=job_seq_Sz1.err
#SBATCH --time=02:00:00

module load stack/.2024-04-silent stack/2024-04 gcc julia openmpi
export JULIA_DEPOT_PATH="$HOME/.julia:/cluster/work/math/hpclab/julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia E-Szi-S2_pll.jl 1