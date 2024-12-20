#!/bin/bash
#SBATCH --uenv=julia/24.9:v1
#SBATCH --view=modules
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --output=job%j.out
#SBATCH --error=job%j.err
#SBATCH --time=00:08:00
#SBATCH -A <Account name>
#SBATCH --partition=normal
#SBATCH --job-name=DMRG_pll_8
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
module load  cray-mpich/8.1.30    hdf5/1.14.3 
export JULIA_DEPOT_PATH="/capstor/scratch/cscs/rgraf/daint/juliaup/depot"
export PATH="$SCRATCH/daint/juliaup/bin:$PATH"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -n 4 julia /users/{username}/Bachelor-Arbeit/src/DMRG_template_pll_Energyextrema.jl 0.5 10 0.4 0 1 true true true > parent_calc_folder/dmrg_.out 2> parent_calc_folder/dmrg_.err
