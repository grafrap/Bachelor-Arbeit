#!/bin/bash
#SBATCH --uenv=julia/24.9:v1
#SBATCH --view=modules
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=10GB
#SBATCH --output=job%j.out
#SBATCH --error=job%j.err
#SBATCH --time=00:10:00
#SBATCH -A <Account name>
#SBATCH --partition=normal
#SBATCH --job-name=Dyncorr
#SBATCH --no-requeue

module load  cray-mpich/8.1.30    hdf5/1.14.3 
export JULIA_DEPOT_PATH="/capstor/scratch/cscs/$USER/daint/juliaup/depot"
export PATH="$SCRATCH/daint/juliaup/bin:$PATH"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
cp *.h5 parent_calc_folder/
/users/$USER/Bachelor-Arbeit/src/Dynam_Corr.jl 1 1000 200 > parent_calc_folder/dyncorr.out 2> parent_calc_folder/dyncorr.err
