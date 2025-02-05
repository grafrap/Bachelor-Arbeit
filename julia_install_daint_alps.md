Installation guide for julia on daint@alps
```shell
uenv repo create
```
```shell
uenv image pull julia
```
(or specific version)
```shell
uenv start --view=modules julia
```
```shell
export JULIA_DEPOT_PATH="$SCRATCH/daint/juliaup/depot"
export PATH="$SCRATCH/daint/juliaup/bin:$PATH"
juliaup add 1.10
```
(let everything download)
```shell
juliaup default 1.10
```
```shell
module load cray-mpich hdf5
```
```shell
julia
```
go to pkg mode
```shell
add HDF5 LinearAlgebra Strided Dates Random InteractiveUtils DelimitedFiles Plots Printf ITensors ITensorMPS MPIPreferences MPI
```
exit to normal terminal
```shell
mkdir ITensorParallel
cd ITensorParallel
git clone https://github.com/ITensor/ITensorParallel.jl.git
```
```shell
cd ..
```
```shell
julia
```
```shell
using Pkg;    Pkg.develop(path="./ITensorParallel/ITensorParallel.jl");    using ITensorParallel
```
go to pkg mode
```shell
precompile
```
In order to run the dyncorr calculation, in the first line, you have to add the path to your julia installation like this:
```shell
which julia
```
copy the output and write it together with `#!` to the first line of the script Dyynam_Corr.jl
make the Dynam_Corr.jl script executable with (go to the src folder for this)
```shell
chmod 755 Dynam_Corr.jl
```
