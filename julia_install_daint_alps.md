Installation guide for julia on daint@alps
```shell
uenv repo create
```
```shell
uenv image pull julia
```
(or specific version)
```shell
uenv start --view julia,modules julia
```
```shell
juliaup init
```
(let everything download)
```shell
module load cray-mpich hdf5
```
```shell
julia
```shell
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
