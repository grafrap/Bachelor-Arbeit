# command for this script: mpiexecjl -n 4 julia DMRG_template_pll.jl 0.5 100 0.4 0 1 true true > outputs/output.txt 2> outputs/error.txt
###############################################################################
# packages #
###############################################################################
using MPIPreferences
MPIPreferences.use_system_binary()
using MPI: MPI
MPI.Init()

using ITensors
using ITensorMPS
using ITensorParallel
using LinearAlgebra: BLAS
using Strided
using Dates
using Random
using HDF5
using InteractiveUtils


BLAS.set_num_threads(1)

lib_dir = "."
include(lib_dir*"/Operators.jl")
include(lib_dir*"/Hamiltonian.jl")
include(lib_dir*"/DMRGSweeps.jl")
import .Operators
import .Hamiltonian
import .DMRGSweeps
# run this script with: mpiexecjl -n 4 julia DMRG_template_pll.jl 0.5 10 0.4 0 1 true true > outputs/output.txt

# functions #
function spinstate_sNSz(s,N,Sz;random="yes")
  #=
       function that creates a spin state from s,N,Sz

       Notes:
       - as of now, only s=1/2 and s=1 are possible
       - for s=1, the state created has minimal nr of "Z0"
       more info on the available spin states: https://itensor.github.io/ITensors.jl/dev/IncludedSiteTypes.html
       more on new s: https://github.com/ITensor/ITensors.jl/blob/main/src/lib/SiteTypes/src/sitetypes/spinhalf.jl
  =#
  # In total adds up to Sz, with available states from -s, -s+1, ..., s
    # TODO: Generalize this function for many s
    # general s
  if abs(Sz) > s*N
    throw(ArgumentError("Check s, N and Sz"))
  end

  state = fill("Dn", N)
  
  # s=1/2
  if s==1/2
    if N%2 == abs(2*Sz)%2
      for i in 1:Int(N/2+Sz)
        state[i] = "Up"
      end
      # example output ["Up", "Dn", "Up", "Up", "Up", "Dn", "Up", "Up", "Dn", "Dn"] for s = 0.5, Sz = 1, N = 10
      # in this case, "Up" means Sz = 1/2 and "Dn" means Sz = -1/2, sum of all sz(i) = Sz
    else
      throw(ArgumentError("Check s, N and Sz"))
    end
  # s=1
  elseif s==1
    if (2*Sz)%2 == 0
      if N%2==0
        for i in 1:Int(N/2+floor(Sz/2))
          state[i] = "Up"
        end
        if abs(Sz%2)==1
          state[Int(N/2+floor(Sz/2))+1] = "Z0"
        end
      elseif N%2==1
        for i in 1:Int(floor(N/2)+ceil(Sz/2))
          state[i] = "Up"
        end
        if abs(Sz%2)==0
          state[Int(floor(N/2)+ceil(Sz/2))+1] = "Z0"
        end
      end
    else
      throw(ArgumentError("Check s, N and Sz"))
    end
    # example output: ["Z0", "Dn", "Up", "Up", "Dn", "Up", "Up", "Up", "Dn", "Dn"] for s = 1, Sz = 1, N = 10
    # in this case, "Up" means Sz = 1, "Dn" means Sz = -1 and "Z0" means Sz = 0, sum of all sz(i) = Sz
  else
    println(stderr, "ERROR: as of now, only s=1/2 and s=1 are possible")
    exit()
  end

  if random=="yes"
    shuffle!(state)
  end

  return(state)
end

# main #
function main()

# start time
start_time = DateTime(now())

# Set the number of threads to 1, to use all cores as MPI processes
BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()

# For now, disable threaded blocksparse
ITensors.enable_threaded_blocksparse(false)
# ITensors.enable_threaded_blocksparse(true)

# Get all needed MPI information
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)

# Get setup, only rank 0 will do this
if rank == 0

  # Check if the correct number of arguments is provided
  if length(ARGS) < 7
    println(stderr, "Usage: julia DMRG_template.jl <s> <N> <J> <Sz> <nexc> <conserve_symmetry> <print_HDF5>")
    exit(1)
  end

  # physical parameters
  s = parse(Float64, ARGS[1])
  N = parse(Int, ARGS[2])
  J1 = parse(Float64, ARGS[3])
  
  # create the J matrix, with J1 on the off-diagonal TODO: generalize for all J matrices
  J = zeros(N,N)
  for i in 1:N-1
    J[i,i+1] = J1 + (-1)^i * 0.03 * J1
    J[i+1,i] = J1
    if i != N-1
      J[i,i+2] = 0.19 *J1
      J[i+2,i] = 0.19 *J1
    end
  end
  
  # other parameters
  Sz = parse(Float64, ARGS[4]) # Sz = 1/2 or 0 for now
  @assert Sz == 1/2 || Sz == 0 || Sz == 1 "Sz must be 1/2 or 0"
  nexc = parse(Int, ARGS[5])
  
  conserve_symmetry, print_HDF5 = true, true
  try
    conserve_symmetry = parse(Bool, ARGS[6])
  catch e
    throw(ArgumentError("Failed to parse conserve_symmetry as Bool: $e"))
  end

  try
    print_HDF5 = parse(Bool, ARGS[7])
  catch e
    throw(ArgumentError("Failed to parse print_HDF5 as Bool: $e"))
  end

  
  # system
  Nsites = N
  if (2*s)%2 == 0
    sites = siteinds("S="*string(Int(s)),Nsites;conserve_sz=conserve_symmetry)
  else
    sites = siteinds("S="*string(Int(2*s))*"/2",Nsites;conserve_sz=conserve_symmetry)
  end
  
  # initial state
  statei = spinstate_sNSz(s, N, Sz)
  linkdim = 100 # variable to randomize initial MPS
  ψi = randomMPS(sites, statei, linkdim)
  
  # S² operator
  S2 = Operators.S2_op(Nsites, sites)
  
  # Sz(i) operator
  Szi = [Operators.Szi_op(i, sites) for i in 1:Nsites]

  # Identity operator
  I = Operators.Identity_op(Nsites, sites)

  # Print the operators to HDF5 file
  if print_HDF5
    h5file = h5open("outputs/operators_data.h5", "w")
    write(h5file, "S2", S2)
    group = create_group(h5file, "Sz")
    for i in 1:Nsites
      write(group, "Sz_$i", Szi[i])
    end
    write(h5file, "I", I)
    close(h5file)
  end
  
  # Hamiltonian
  H = Hamiltonian.H(N, sites, J=J)

  # Print the Hamiltonian to HDF5 file
  if print_HDF5
    h5file = h5open("outputs/H_data.h5", "w")
    write(h5file, "H", MPO(H, sites))
    close(h5file)
  end

  # Partition the Hamiltonian for parallel DMRG
  Hpart = partition(H, nprocs)

else
  # empty variables for other processes

  sites = nothing
  ψi = nothing
  S2 = nothing
  Szi = nothing
  Sz = nothing
  H = nothing
  Hpart = nothing
  nexc = nothing
  Nsites = nothing
  print_HDF5 = nothing
end

# Broadcast the variables from rank 0 to all processes
sites = ITensorParallel.bcast(sites, 0, MPI.COMM_WORLD)
ψi = ITensorParallel.bcast(ψi, 0, MPI.COMM_WORLD)
S2 = ITensorParallel.bcast(S2, 0, MPI.COMM_WORLD)
Szi = ITensorParallel.bcast(Szi, 0, MPI.COMM_WORLD)
Sz = ITensorParallel.bcast(Sz, 0, MPI.COMM_WORLD)
H = ITensorParallel.bcast(H, 0, MPI.COMM_WORLD)
Hpart = ITensorParallel.bcast(Hpart, 0, MPI.COMM_WORLD)
nexc = ITensorParallel.bcast(nexc, 0, MPI.COMM_WORLD)
Nsites = ITensorParallel.bcast(Nsites, 0, MPI.COMM_WORLD)
print_HDF5 = ITensorParallel.bcast(print_HDF5, 0, MPI.COMM_WORLD)

# DMRG precision parameters
precE = 1E-8
precS2 = 1E-8
precSzi = 1E-8
w = 1E5 # penalty for non-orthogonality

# sum term for parallel DMRG
MPI.Barrier(MPI.COMM_WORLD)
mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
MPI.Barrier(MPI.COMM_WORLD)

# ground state
E0, ψ0 = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi)

# save ground state
En = [E0]
ψn = [ψ0]

# calculate the highest energy state with -H 
MPI.Barrier(MPI.COMM_WORLD)
mpo_sum_term = MPISumTerm(MPO(-Hpart[rank+1], sites), MPI.COMM_WORLD)
MPI.Barrier(MPI.COMM_WORLD)

# highest energy state
E1, ψ1 = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi)

# save highest energy state
En = [E0, E1]


if rank == 0
  # <ψn|S²|ψn>
  S2n = [inner(ψn[i]',S2,ψn[i]) for i in eachindex(ψn)]
  
  # <ψn|Sz(i)|ψn>
  Szin = [[inner(ψn[i]',Szi[j],ψn[i]) for j in 1:Nsites] for i in eachindex(ψn)]
  
  time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now()) - Dates.DateTime(start_time)))

  mkpath("outputs")

  fw = open("outputs/Szi_Sz_pll=$(Sz).txt", "w")
  # outputs
  write(fw, "List of E:\n")
  write(fw, string(En), "\n")
  write(fw, "\n")
  write(fw, "List of S²:\n")
  write(fw, string(S2n), "\n")
  write(fw, "\n")
  write(fw, "List of Sz(i):\n")
  write(fw, string(Szin), "\n")
  write(fw, "----------\n\n")
  write(fw,"total time = "* string(time) * "\n")
  close(fw)

  @show time
  
  if print_HDF5
  
    println(stderr, "Printing to HDF5 file")

    # Open an HDF5 file for writing
    h5file = h5open("outputs/psin_data.h5", "w")
    
    # Write the MPS data to the HDF5 file
    for (i, psi) in enumerate(ψn)
      group = create_group(h5file, "state_$i")
      write(group, "ψ", psi)
    end
    
    # Close the HDF5 file
    close(h5file) 
  end

end
end

main()
MPI.Finalize()
