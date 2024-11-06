#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=120GB
#SBATCH --output=daint/chi_6000_om100.out
#SBATCH --error=daint/dynam_corr_6000_om100.err
#SBATCH --time=24:00:00
#SBATCH -A em01
#SBATCH -C mc
#SBATCH --partition=normal
#SBATCH --job-name=Dynam_corr_6000_om100

module load daint-mc JuliaExtensions 
export JULIA_DEPOT_PATH="$HOME/.julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun julia -t 9 Dynam_Corr.jl 1000
