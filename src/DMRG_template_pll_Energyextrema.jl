# command for this script: mpiexecjl -n 4 julia DMRG_template_pll.jl 0.5 100 0.4 0 1 true true > outputs/output.txt 2> outputs/error.txt
###############################################################################
# packages #
###############################################################################
# src/DMRG_template_pll.jl
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
include(lib_dir*"/Customspace.jl")
import .Operators
import .Hamiltonian
import .DMRGSweeps
# run this script with: mpiexecjl -n 4 julia DMRG_template_pll_Energyextrema.jl 0.5 10 0.4 0 0 true true > outputs/output.txt (julia 1.9)
# run this script with: mpirun -n 4 julia DMRG_template_pll_Energyextrema.jl 0.5 10 0.4 0 0 false true > outputs/output.txt (julia 1.10/1.11)
# Function to parse arguments
function parse_arguments()
  input_args = []
  if length(ARGS) == 0
      # Read a line from stdin
      println("No command-line arguments provided. Reading from stdin...")
      try
          input_line = readline(stdin)
          # Remove surrounding quotes if present
          input_line = strip(input_line, ['"', '\''])
          # Split the input line into arguments
          input_args = split(input_line)
          println("Arguments read from stdin: ", input_args)
      catch e
          @error "Failed to read from stdin." exception=(e, catch_backtrace())
          exit(1)
      end
  else
      input_args = ARGS
  end

  if length(input_args) != 8
      println("Usage: julia DMRG_template_pll_Energyextrema.jl <s> <N> <J> <Sz> <nexc> <conserve_symmetry> <print_HDF5> <maximal_energy>")
      exit(1)
  end

  # Parse arguments
  s = parse(Float64, input_args[1])
  N = parse(Int, input_args[2])
  J = parse(Float64, input_args[3])
  Sz = parse(Int, input_args[4])
  nexc = parse(Int, input_args[5])
  conserve_symmetry = parse(Bool, input_args[6])
  print_HDF5 = parse(Bool, input_args[7])
  maximal_energy = parse(Bool, input_args[8])

  return (s, N, J, Sz, nexc, conserve_symmetry, print_HDF5, maximal_energy)
end



# functions #
function spinstate_sNSz(s,N,Sz;random="yes")
  #=
      function that creates a spin state from s,N,Sz

      Notes:
      - as of now, only s=1/2 and s=1 are possible
      - for s=1, the state created has minimal nr of "Z0"
      more info on the available spin states: https://itensor.github.io/ITensors.jl/dev/IncludedSiteTypes.html
      more on new s: https://github.com/ITensor/ITensors.jl/blob/main/src/lib/SiteTypes/src/sitetypes/spinhalf.jl
      Instructions on new sitetype: https://itensor.github.io/ITensors.jl/stable/examples/Physics.html
  =#
  # In total adds up to Sz, with available states from -s, -s+1, ..., s
  # general s
  # Check if Sz is valid
  if abs(Sz) > s * N
    @error "Check s, N and Sz"
    exit(1)
  end
  # Check if Sz is reachable with the given s
  if (s*N) % 1 != Sz % 1
    @error "Check s, N and Sz"
    exit(1)
  end

  s = Rational(s)
  # Get the possible states for the given spin value
  possible_states = generate_spin_states(s)
  state = fill(possible_states[1], N)
  sum = N * -s
  state_index_map = state_to_index(s)

  # Fill the state array with the correct states
  while sum < Sz
    for i in 1:N
      if sum == Sz
        break
      end
      # Check if current state is already the biggest state
      if state_index_map[state[i]] == length(possible_states)
        continue
      end
      # Check, if we are free to choose a random bigger state
      if sum + s < Sz
        # Assign random bigger state to the current index
        current_state = possible_states[rand(state_index_map[state[i]]+1:end)]
        state[i] = current_state
        sum += state_index_map[current_state] - 1
      else
        # Assign the restvalue to the current index
        state[i] = possible_states[Int(state_index_map[state[i]] + (Sz - sum) + 1)]
        sum += state_index_map[state[i]]-1
      end
    end
  end

  # Shuffle the state array if randomization is requested
  if random == "yes"
    shuffle!(state)
  end

  index_vector = Vector{Int}(undef, length(state))
  for i in eachindex(state)
    index_vector[i] = state_index_map[state[i]]
  end

  return index_vector
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
    
  s, N, J1, Sz, nexc, conserve_symmetry, print_HDF5, maximal_energy = parse_arguments()
  println(stderr, "Parameters:")
  println(stderr, "s = $s, N = $N, J = $J1, Sz = $Sz, nexc = $nexc, conserve_symmetry = $conserve_symmetry, print_HDF5 = $print_HDF5, maximal_energy = $maximal_energy")
  # create the J matrix, with J1 on the off-diagonal TODO: generalize for all J matrices
  J = zeros(N,N)
  for i in 1:N-1
    J[i,i+1] = J1 + (-1)^i * 0.03 * J1
    J[i+1,i] = J1 + (-1)^i * 0.03 * J1
    if i != N-1
      J[i,i+2] = 0.19 *J1
      J[i+2,i] = 0.19 *J1
    end
  end
  
  
  # TODO: check if there are more symmetry conservation options
  # https://github.com/ITensor/ITensors.jl/blob/f37d4a5dd8a7376d0daedd74bc326cb6f9653b00/src/lib/SiteTypes/src/sitetypes/spinhalf.jl#L2-L13
  # available options: conserve_qns, conserve_sz, conserve_szparitiy
  # system
  Nsites = N
  
  if (2*s)%2 == 0
    sites = siteinds("S="*string(Int(s)),Nsites;conserve_sz=conserve_symmetry)
  elseif (2*s)%2 == 1
    sites = siteinds("S="*string(Int(2*s))*"/2",Nsites;conserve_sz=conserve_symmetry)
  else
    @error "Check s"
    exit(1)
  end
  
  # initial state
  statei = spinstate_sNSz(s, N, Sz)
  linkdim = 100 # variable to randomize initial MPS$
  ψi = randomMPS(sites, statei; linkdims=linkdim)
  
  # S² operator
  S2 = Operators.S2_op(Nsites, sites)
  
  # Sz(i) operator
  Szi = [Operators.Szi_op(i, sites) for i in 1:Nsites]

  # Print the operators to HDF5 file
  if print_HDF5
    h5file = h5open("outputs/operators_data.h5", "w")
    write(h5file, "sites", sites)
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
precE = 1E-6
precS2 = 1E-6
precSzi = 1E-6
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

# excited states
for _ in 1:nexc
  MPI.Barrier(MPI.COMM_WORLD)
  mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
  MPI.Barrier(MPI.COMM_WORLD)
  E,ψ = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term,ψi,precE,precS2,precSzi,S2,Szi,ψn=ψn,w=w)
  push!(En,E)
  push!(ψn,ψ)
end


if maximal_energy
  # calculate the highest energy state with -H 
  MPI.Barrier(MPI.COMM_WORLD)
  mpo_sum_term = MPISumTerm(MPO(-Hpart[rank+1], sites), MPI.COMM_WORLD)
  MPI.Barrier(MPI.COMM_WORLD)

  # highest energy state
  E1, ψ1 = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi)

  # save highest energy state
  push!(En, E1)
  push!(ψn, ψ1)
end


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
