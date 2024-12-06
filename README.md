First install julia or make sure you got Julia version 1.10 or higher:
curl -fsSL https://install.julialang.org | sh
Then create environment
cd ~
mkdir julia; cd julia
julia
]
activate DMRGenv
add HDF5 LinearAlgebra Strided Dates Random InteractiveUtils DelimitedFiles Plots Printf ITensors ITensorMPS MPIPreferences MPI
backspace
exit()
mkdir ITensorParallel
cd ITensorParallel
git clone https://github.com/ITensor/ITensorParallel.jl.git
cd ..
julia
]
activate DMRGenv
backspace
using Pkg;    Pkg.develop(path="./ITensorParallel/ITensorParallel.jl");    using ITensorParallel
]
precompile
instantiate

run from now on: julia --project=~/julia/DMRGenv script.jl

install the DMRG code: https://github.com/grafrap/Bachelor-Arbeit.git
Make sure you installed also a version of mpi. The path to the mpi lib should also be added to the ~/.bashrc or ~/.zshrc file on Linux / MacOS respectively.

Install the code on your machine for ghe usage of the aiida workchain
verdi code create core.code.installed --label dmrg --computer localhost --filepath-executable /home/jovyan/Bachelor-Arbeit/src/DMRG_template_pll_Energyextrema.jl --non-interactive

verdi code list 
cd aiida-dmrg 
pip install --user -e .
verdi plugin list aiida.calculations
