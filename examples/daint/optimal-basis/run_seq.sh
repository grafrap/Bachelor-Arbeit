#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --output=job_seq_Sz0.out
#SBATCH --error=job_seq_Sz0.err
#SBATCH --time=04:00:00
#SBATCH -A em01
#SBATCH -C mc
#SBATCH --partition=normal
#SBATCH --job-name=DMRG_seq

module load daint-mc JuliaExtensions 
export JULIA_DEPOT_PATH="$HOME/.julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia E-Szi-S2_pll.jl 0