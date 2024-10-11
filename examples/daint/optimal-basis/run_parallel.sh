#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=job_Sz0_pll_32.out
#SBATCH --error=job_Sz0_pll_32.err
#SBATCH --time=02:00:00
#SBATCH --account=em01
#SBATCH --constraint=mc
#SBATCH --partition=normal
#SBATCH --job-name=DMRG_parallel_32

module load stack/.2024-04-silent stack/2024-04 gcc julia openmpi
export JULIA_DEPOT_PATH="$HOME/.julia:/cluster/work/math/hpclab/julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia E-Szi-S2_pll.jl 0