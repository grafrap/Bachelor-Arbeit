#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --output=job_Sz0_pll_8.out
#SBATCH --error=job_Sz0_pll_8.err
#SBATCH --time=04:00:00
#SBATCH -A em01
#SBATCH -C mc
#SBATCH --partition=normal
#SBATCH --job-name=DMRG_pll_8

module load daint-mc JuliaExtensions 
export JULIA_DEPOT_PATH="$HOME/.julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -n 8 julia E-Szi-S2_pll.jl 0