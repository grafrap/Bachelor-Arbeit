#!/bin/bash

# Load necessary modules
module load daint-mc JuliaExtensions 

# Extract the registry archive
# unzip General.zip
# mkdir -p ~/.julia/registries
# mv General-master ~/.julia/registries/General
# Run Julia and add MPI package
julia -e 'using Pkg; Pkg.add("MPI");   Pkg.add("ITensors"); Pkg.add("ITensorMPS"); Pkg.add("Dates"); Pkg.add("LinearAlgebra"); Pkg.add("Strided");'

# install ITensorParallel: 
# git clone https://github.com/ITensor/ITensorParallel.jl.git into a folder
# in the julia REPL: using Pkg;    Pkg.develop(path="./ITensorParallel.jl");    using ITensorParallel