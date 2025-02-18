# command for this script: mpirun -n 4 julia DMRG_template_pll_Energyextrema.jl 0.5 100 0.4 0 1 true true true > outputs/output.txt 2> outputs/error.txt
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

# Function to parse arguments
function parse_arguments()
  input_args = []
  
  if length(ARGS) == 0
    # Read a line from stdin
    println("No command-line arguments provided. Reading from input file...")
    try
      input_line = readline(stdin)
      # Remove surrounding quotes if present
      input_line = strip(input_line, ['"', '\''])
      # Split the input line into tokens
      tokens = split(input_line)
    catch e
      @error "Failed to read from stdin." exception=(e, catch_backtrace())
      MPI.Abort(MPI.COMM_WORLD, 1)
    end
  else
    tokens = ARGS
  end

  # Process tokens to handle the J matrix as a single argument
  i = 1
  while i <= length(tokens)
    token = tokens[i]
    if startswith(token, "[[")
      # Start collecting J matrix tokens
      j_tokens = [token]
      i += 1
      while i <= length(tokens) && !endswith(tokens[i], "]]")
        push!(j_tokens, tokens[i])
        i += 1
      end
      if i <= length(tokens)
        push!(j_tokens, tokens[i])  # Add the closing brackets
        i += 1
      else
        @error "J matrix not properly closed with ']]'."
        MPI.Abort(MPI.COMM_WORLD, 1)
      end
      # Join all J tokens into one string
      J_str = join(j_tokens, " ")
      # Convert Python-like array to Julia-like array
      J_str = strip(J_str, ['[', ']'])
      J_str = replace(J_str, r"\],\s*\[" => "; ")
      J_str = replace(J_str, "," => " ")
      J_str = "[" * J_str * "]"
      push!(input_args, J_str)      

    elseif startswith(token, "[")
      # Start collecting J matrix tokens
      j_tokens = [token]
      i += 1
      while i <= length(tokens) && !endswith(tokens[i], "]")
        push!(j_tokens, tokens[i])
        i += 1
      end
      if i <= length(tokens)
        push!(j_tokens, tokens[i])  # Add the closing bracket
        i += 1
      else
        @error "J matrix not properly closed with ']'."
        MPI.Abort(MPI.COMM_WORLD, 1)
      end
      # Join all J tokens into one string
      J_str = join(j_tokens, " ")
      push!(input_args, J_str)
    else
      # Regular argument
      push!(input_args, token)
      i += 1
    end
  end

  if length(input_args) != 9 && length(input_args) != 8
    println("Usage: julia DMRG_template_pll_Energyextrema.jl <s> <N> <J> <Sz> <nexc> <conserve_symmetry> <print_HDF5> <maximal_energy>")
    MPI.Abort(MPI.COMM_WORLD, 1)
  end

  # Parse arguments
  s = parse(Float64, input_args[1])
  N = parse(Int, input_args[2])
  cutoff = parse(Float64, input_args[3])
  J_input = input_args[4]
  
  # Initialize J as Union{Float64, Matrix{Float64}}
  J = nothing

  # Check if J is a matrix and parse it
  if startswith(J_input, "[") && endswith(J_input, "]")
    # Parse J as a matrix
    J_content = J_input[2:end-1]  # Remove surrounding brackets
    rows = split(J_content, ";")
    try
      # Parse each row into a vector of Float64
      J_matrix = [parse.(Float64, split(strip(row))) for row in rows]
      # Ensure all rows have the same number of columns
      row_lengths = [length(row) for row in J_matrix]
      if length(unique(row_lengths)) != 1
        @error "All rows in J matrix must have the same number of columns."
        MPI.Abort(MPI.COMM_WORLD, 1)
      end
      # Convert arrays of arrays to a 2D Matrix
      J = reduce(vcat, J_matrix)
      J = reshape(J, row_lengths[1], length(J_matrix))
    catch e
      @error "Failed to parse J matrix: $e"
      MPI.Abort(MPI.COMM_WORLD, 1)
    end
  else
    # Parse J as a scalar
    try
      J = parse(Float64, J_input)
    catch e
      @error "Failed to parse J as a Float64: $e"
      MPI.Abort(MPI.COMM_WORLD, 1)
    end
  end

  # Parse the remaining arguments
  if length(input_args) == 8
    Sz = nothing
    nexc = parse(Int, input_args[5])
    conserve_symmetry = parse(Bool, input_args[6])
    print_HDF5 = parse(Bool, input_args[7])
    maximal_energy = parse(Bool, input_args[8])
  else
    Sz = parse(Float64, input_args[5])
    nexc = parse(Int, input_args[6])
    conserve_symmetry = parse(Bool, input_args[7])
    print_HDF5 = parse(Bool, input_args[8])
    maximal_energy = parse(Bool, input_args[9])
  end

  # Check if nothing in Sz is valid
  if Sz === nothing && conserve_symmetry == true
    @error "Sz must be provided when conserve_symmetry is true."
    MPI.Abort(MPI.COMM_WORLD, 1)
  end

  # Set Sz to nothing, because it is not needed
  if conserve_symmetry == false
    Sz = nothing
  end

  return (s, N, cutoff, J, Sz, nexc, conserve_symmetry, print_HDF5, maximal_energy)
end

# functions #
function spinstate_sNSz(s, N, Sz; random="yes")
  #=
      function that creates a spin state from s, N, Sz (Sz only if conserve_symmetry is true)
      In total states add up to Sz, with available states from -s, -s+1, ..., s
  =#

  # Get the possible states for the given spin value
  s = Rational(s)
  possible_states = generate_spin_states(s)
  state_index_map = state_to_index(s)
  state = fill(possible_states[1], N)
  index_vector = Vector{Int}(undef, length(state))

  # initialize index vector to some random values if Sz is not provided (i.e. no symmetry conservation)
  if Sz === nothing
    random_ind = rand(1:Int(2*s+1), N)
    for i in 1:N
      state[i] = possible_states[random_ind[i]]
      index_vector[i] = state_index_map[state[i]]
    end
    return index_vector
  end

  # Check if Sz is valid
  if abs(Sz) > s * N
    @error "Check s, N and Sz"
    MPI.Abort(MPI.COMM_WORLD, 1)
  end
  # Check if Sz is reachable with the given s
  if (s * N) % 1 != Sz % 1
    @error "Check s, N and Sz"
    MPI.Abort(MPI.COMM_WORLD, 1)
  end

  # Initialize the sum of the states to compare with Sz
  sum = N * -s

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
      if sum - state_index_map[state[i]] + 1 + 2*s < Sz
        # Assign random bigger state to the current index
        current_state = possible_states[rand(state_index_map[state[i]]+1:end)]
        state[i] = current_state
        sum += state_index_map[current_state] - 1
      else
        # Assign the rest value to the current index
        state[i] = possible_states[Int(state_index_map[state[i]] + (Sz - sum))]
        sum += state_index_map[state[i]] - 1
      end
    end
  end

  # Shuffle the state array if randomization is requested
  if random == "yes"
    shuffle!(state)
  end

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

    s, N, cutoff, J_input, Sz, nexc, conserve_symmetry, print_HDF5, maximal_energy = parse_arguments()
    println("Parameters:")
    println("s = $s, N = $N, cutoff = $cutoff, J = $J_input, Sz = $Sz, nexc = $nexc, conserve_symmetry = $conserve_symmetry, print_HDF5 = $print_HDF5, maximal_energy = $maximal_energy")

    # Determine if J is scalar or matrix
    if typeof(J_input) == Float64
      # Create the J matrix with J1 on the off-diagonal
      J = zeros(N, N)
      for i in 1:N-1
        J[i, i+1] = J_input
        J[i+1, i] = J_input
      end
    elseif typeof(J_input) == Matrix{Float64}
      # Use the provided J matrix
      J = J_input
      # Optional: Validate the J matrix dimensions
      if size(J, 1) != N || size(J, 2) != N
        @error "J matrix dimensions ($(size(J, 1))x$(size(J, 2))) do not match N=$N."
        MPI.Abort(MPI.COMM_WORLD, 1)
      end
    else
      @error "J must be either a Float64 scalar or a Matrix{Float64}."
      MPI.Abort(MPI.COMM_WORLD, 1)
    end

    # system
    Nsites = N

    if (2 * s) % 2 == 0
      sites = siteinds("S="*string(Int(s)), Nsites; conserve_sz=conserve_symmetry)
    elseif (2 * s) % 2 == 1
      sites = siteinds("S="*string(float(s)), Nsites; conserve_sz=conserve_symmetry)
    else
      @error "Check s"
      MPI.Abort(MPI.COMM_WORLD, 1)
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
      h5file = h5open("operators_data.h5", "w")
      write(h5file, "sites", sites)
      close(h5file)
    end

    # Hamiltonian
    H = Hamiltonian.H(N, sites, J=J)

    # Print the Hamiltonian to HDF5 file
    if print_HDF5
      h5file = h5open("H_data.h5", "w")
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
    maximal_energy = nothing
    cutoff = nothing
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
  maximal_energy = ITensorParallel.bcast(maximal_energy, 0, MPI.COMM_WORLD)
  cutoff = ITensorParallel.bcast(cutoff, 0, MPI.COMM_WORLD)

  # DMRG precision parameters
  precE = 1E-6
  precS2 = 1E-4
  precSzi = 1E-4
  w = 1E5 # penalty for non-orthogonality

  # sum term for parallel DMRG
  MPI.Barrier(MPI.COMM_WORLD)
  mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
  MPI.Barrier(MPI.COMM_WORLD)

  # ground state
  E0, ψ0 = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi, cutoff=cutoff)

  # save ground state
  En = [E0]
  ψn = [ψ0]

  # excited states
  for excit in 1:nexc
    if rank == 0
      println("Calculating excited state $excit")
    end
    # sum term for parallel DMRG
    MPI.Barrier(MPI.COMM_WORLD)
    mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
    MPI.Barrier(MPI.COMM_WORLD)

    # excited state calculation
    E, ψ = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi, cutoff=cutoff, ψn=ψn, w=w)
    
    # save excited state
    push!(En, E)
    push!(ψn, ψ)
  end

  # calculate the highest energy state with -H 
  if maximal_energy
    if rank == 0
      println("Calculating highest energy state")
    end
    MPI.Barrier(MPI.COMM_WORLD)
    mpo_sum_term = MPISumTerm(MPO(-Hpart[rank+1], sites), MPI.COMM_WORLD)
    MPI.Barrier(MPI.COMM_WORLD)

    # highest energy state calculation
    E1, ψ1 = DMRGSweeps.DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi, cutoff=cutoff)

    # save highest energy state
    push!(En, E1)
    push!(ψn, ψ1)
  end

  if rank == 0
    # <ψn|S²|ψn>
    S2n = [inner(ψn[i]', S2, ψn[i]) for i in eachindex(ψn)]

    # <ψn|Sz(i)|ψn>
    Szin = [[inner(ψn[i]', Szi[j], ψn[i]) for j in 1:Nsites] for i in eachindex(ψn)]

    time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now()) - Dates.DateTime(start_time)))

    # outputs
    println("Cutoff = ", cutoff)
    println("List of E:")
    println(En)
    println()
    println("List of S²:")
    println(S2n)
    println()
    println("List of Sz(i):")
    println(Szin)
    println("----------")
    println()
    println("total time = ", time)

    if print_HDF5

      println("Printing to HDF5 file")

      # Open an HDF5 file for writing
      h5file = h5open("psin_data.h5", "w")
      
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