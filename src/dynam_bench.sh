#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --output=daint/benchmark/bench.out
#SBATCH --error=daint/benchmark/bench.err
#SBATCH --time=24:00:00
#SBATCH -A em01
#SBATCH -C mc
#SBATCH --partition=normal
#SBATCH --job-name=benchmark_dynam_corr

module load daint-mc JuliaExtensions 
export JULIA_DEPOT_PATH="$HOME/.julia:$JULIA_DEPOT_PATH"

export PATH="$HOME/.julia/bin:$PATH"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
for i in {1..20}
do
  srun julia -t 8 Dynam_Corr.jl $((50 * i)) > "daint/benchmark/bench_$((50 * i)).out" 2> "daint/benchmark/bench_$((50 * i)).err"
done