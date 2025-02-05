# Parallelized DMRG
## Basic installations
First install julia or make sure you got Julia version 1.10 or higher:
```shell
curl -fsSL https://install.julialang.org | sh
```
Then create environment
```shell
cd ~
mkdir julia; cd julia
julia
]
activate DMRGenv
add HDF5 LinearAlgebra Strided Dates Random InteractiveUtils DelimitedFiles Plots Printf ITensors ITensorMPS MPIPreferences MPI
```
Go back to the Julia REPL with backspace
```shell
exit()
mkdir ITensorParallel
cd ITensorParallel
git clone https://github.com/ITensor/ITensorParallel.jl.git
cd ..
julia
]
activate DMRGenv
```
Go back to Julia REPL with backspace
```shell
using Pkg;    Pkg.develop(path="./ITensorParallel/ITensorParallel.jl");    using ITensorParallel
]
precompile
instantiate
```
If you want to use the environment, run from now on: 
```shell
julia --project=~/julia/DMRGenv script.jl
```
where after the equal sign you add the path to your environment.

Now, also make sure that you [installed a version of MPI](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html). The path to the mpi lib folder should be added to the ~/.bashrc or ~/.zshrc file on Linux / MacOS respectively.

## How to use the code
### DMRG
#### Command
```shell
cd src
mkdir parent_calc_folder
mpirun -n <n> julia --project=~/path-to-env \
DMRG_template_pll_Energyextrema.jl <s> <N> <J> [Sz] \
<nexc> <conserve_symmetry> <print_HDF5> <maximal_energy>\
> parent_calc_folder/dmrg.out 2> parent_calc_folder/dmrg.err
```
or via an input file containing all the arguments:
```shell
mpirun -n <n> julia --project=~/path-to-env \
DMRG_template_pll_Energyextrema.jl < dmrg.inp \
> parent_calc_folder/dmrg.out 2> parent_calc_folder/dmrg.err
```
#### Arguments
`n` [Int]: Number of MPI-processes, usually something between 1 and 8 is optimal, depending on the system and its size.\
`s` [Int] or [Int/2]: spin quantum number of every spin in the chain.\
`N` [Int]: length of the spin chain.\
`J` [Float64] or [Matrix{Float64}] of size N x N: Value / matrix containing all the J values for the Heisenberg term of the Hamiltonian.\
`Sz` [Int] or [Int/2]: quantum number defining the sum of the local $S_z$ quantum number over the whole chain. This parameter is only important if `conserve_symmetry` is set to true, else `Sz` gets automatically set to nothing. It is therefore also only necessary if you want to conserve symmetry in your system.\
`nexc` [Int]: Defines the number of excited states which are calculated. If you only want the ground state to be calculated, set it to 0.\
`conserve_symmetry` [Bool]: Defines if the $S_z$ quantum number should be conserved in the calculation.\
`print_HDF5` [Bool]: Defines if you want to print the HDF5 representations for the Hamiltonian, the wave functions and information about the sites. This argument has to be set to true if you want to do a dynamical correlator calculation afterwards.\
`maximal_energy` [Bool]: Defines if the maximal energy of the system should be calculated. This argument has to be set to true if you want to do a dynamical correlator calculation afterwards.
#### Outputs
The script outputs an array of energy values for all calculated states of the Hamiltonian. In addition, it prints the arrays for <ψn|S²|ψn> and <ψn|Sz(i)|ψn>, where ψn are all the calculated wave functions. If specified, it outputs the HDF5 files of the wave function MPS, the Hamiltonian MPO and the sites information.
#### Example
```shell
mpirun -n 4 julia --project=~/julia/DMRGenv \
DMRG_template_pll_Energyextrema.jl 0.5 10 0.4 0 1 true \
true true > parent_calc_folder/dmrg.out 2> parent_calc_folder/dmrg.err
```

### Dynamical Correlator
#### Command
First, make sure, that you copy all created HDF5 files into the folder `parent_calc_folder`
```shell
cp *.h5 parent_calc_folder/
```
```shell
julia -t <t> --project=/path-to-env Dynam_Corr.jl <J> [N] [cutoff] \ 
> dyncorr.out 2> dyncorr.err
```
or with an input file:
```shell
julia -t <t> --project=/path-to-env Dynam_Corr.jl < dyncorr.inp \
> dyncorr.out 2> dyncorr.err
```
#### Arguments
`t` [Int]: Specifies the number of threads for the script. Here more than the number of sites is not necessary nor practical.\
`J` [Float64] or [Matrix{Float64}] of size N x N: Value / matrix containing all the J values for the Heisenberg term of the Hamiltonian. This term should be the same as for the preceding DMRG calculation. It defines the energy range for the $\omega$-vaules of the Dynamical Correlator. Per default the energy range is set to 2 * J if J is a Float64 and 2 * J_mean if J is a matrix, where J_mean is the mean of all nonzero values in the J-matrix. This value is only needed for defining the energy range, hence, if other energy ranges are needed in the calculations, this value can be adjusted.\
`N` [Int]: [Int]: Specifies the number of Chebyshev expansion coefficients. If no N is specified, the default value of max(W, 600) is taken, where W is the effective bandwidth of the Hamiltonian, i.e. the difference between the minimal and maximal energy.\
`cutoff` [Float64]: Defines the cutoff value for the addition function of MPSs. The bigger the value, the faster but the more inaccurate the calculation is. This value is by default set to 1e-8. A reasonable range of cutoff values is between 1e-8 and 1e-6.

From the arguments `N` and `cutoff` none, one or both can be provided. The order of these two does not matter.

#### Outputs
This script outputs the dynamical correlator for ω values between 0 and 2 J_mean (or the given J value) for all sites of the spinchain. Additionally, it already creates a normalized plot of the values.

#### Example
```shell
julia -t 5 --project=~/julia/DMRGenv Dynam_Corr.jl 0.4 1000 1e-6 > \
dyncorr.out 2> dyncorr.err
```

### Installation on daint@alps
Using the information in [julia_install_daint_alps.md](julia_install_daint_alps.md), you can install the code of this repo on daint@alps.

## AiiDA workchain
The DMRG code can be used via an [AiiDAlab workchain](https://github.com/grafrap/aiida-dmrg.git).

```shell
git clone https://github.com/grafrap/aiida-dmrg.git
cd aiida-dmrg 
pip install --user -e .
verdi plugin list aiida.calculations
verdi computer setup --config daint-julia.yaml
verdi code setup --config dmrg-daint-alps.yaml
verdi code setup --config dyncorr-daint-alps.yaml
```
Further information how to use the workchain can be found in the Readme of the aiida-dmrg git repository.
