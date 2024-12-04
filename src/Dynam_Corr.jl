using ITensors
using ITensorMPS
using HDF5
using InteractiveUtils
using LinearAlgebra
using Base.Threads
using Dates

lib_dir = "."
include(lib_dir*"/Operators.jl")
include(lib_dir*"/Customspace.jl")
include(lib_dir*"/Hamiltonian.jl")
import .Operators
import .Hamiltonian


function Jackson_dampening(N::Int)
  #=
    Implements the Jackson dampening factors g_n
  =#
  g = zeros(N+1)
  πN = π/(N+1)
  for n in 1:(N+1)
    g[n] = ((N - n + 1)*cos(πN*n) + sin(πN*n)/tan(πN)) / (N + 1)
  end
  return g
end

function Chebyshev_vectors(B::MPO, H::MPO, ψ::MPS, N::Int)
  #=
    Implements the Chebyshev vectors ∣t_n⟩ = T_n(Ĥ')*Ĉ|ψ⟩
  =#
  t = Vector{MPS}(undef, N+1)
  t[1] = apply(B, ψ; cutoff=1e-8)
  t[2] = apply(H, t[1]; cutoff=1e-8)

  for n in 3:N+1
    t[n] = 2*apply(H, t[n-1]; cutoff=1e-8)
    t[n] = add(t[n], -t[n-2]; cutoff=1e-8)
  end

  return t
end

function Chebyshev_moments(ψ::MPS, A::MPO, t::Array{MPS, 1})
  #=
    Implements the Chebyshev moments μ_n = ⟨ψ|Â*T_n(Ĥ')*B^|ψ⟩ = ⟨ψ|Â∣t_n⟩
  =#
  μ = zeros(length(t))

  for n in 1:length(t)
    μ[n] = inner(ψ', A, t[n])
  end

  return μ
end

function Chebyshev_expansion(x::Float64, N::Int)
  #=
    Implements the Chebyshev expansion for the dynamical correlator
  =# 
  T = zeros(N+1)
  T[1] = 1
  T[2] = x

  for n in 3:N+1
    T[n] = 2*x*T[n-1] - T[n-2]
  end

  return T
end

function Chebyshev_expansion(x::Vector, N::Int)
  #=
    Implements the Chebyshev expansion for the dynamical correlator
  =# 
  T = zeros(size(x)[1], N+1)
  T[:, 1] = ones(size(x))
  T[:, 2] = x

  for n in 3:N+1
    T[:, n] = 2*x.*T[:, n-1] .- T[:, n-2]
  end

  return T
end

function Dynamic_correlator(ψ::MPS, H::MPO, A::MPO, i::Int, I::MPO, E0::Float64, E1::Float64, ω::Vector, N_max; cutoff::Float64=1e-8)
  #=
  Implements the dynamical spin correlator for a vector of frequencies
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#
  
  # check the input
  N_min = 10

  for idx in 1:length(ω)
    @assert ω[idx] < -E0 - E1 "ω must be less than W"
    @assert ω[idx] > 0 "ω must be positive"
  end
  
  # parameters
  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  L = length(ψ)

  H_scaled = add(1/a * add(H, -E0*I; cutoff=cutoff), -W_*I; cutoff=cutoff)

  ω_ = ω./a .- W_ # ω', scaled ω
  Δω = ω[2] - ω[1] # frequency step for integration
  sum_old = 0.0 # integration sum

  #calculate the Jackson dampening factors
  g = Jackson_dampening(N_min)

  # calculate the Chebyshev vectors
  t = Chebyshev_vectors(A, H_scaled, ψ, N_min)

  # calculate the Chebyshev moments
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  χ = zeros(length(ω))

  # build vector for convergence check
  χ_next = zeros(length(ω))
  
  # calculate the Chebyshev expansion of the scaled ω
  T = Chebyshev_expansion(ω_, N_min)

  # calculate the χ for the initial N_min
  prefactor = 1 ./(a * π * sqrt.(1 .- ω_.^2))
  sumval = sum(reduce(hcat, [g[n].*μ[n].*T[:, n] for n in 2:N_min]), dims=2)
  χ = prefactor .* (g[1] * μ[1] .+ 2 .* sumval)

  while true 
    start_time = DateTime(now())

    N_min += 1
    # calculate the Jackson dampening factors for the new N_min
    g = Jackson_dampening(N_min)

    # calculate the Chebyshev vectors for the new N_min
    # It's done this way to make use of cutoff, so that the bond dimensions don't explode
    t_next = 2*apply(H_scaled, t[end]; cutoff=cutoff)
    t_next = add(t_next, -t[end-1]; cutoff=cutoff)

    push!(t, t_next)
    push!(μ, inner(ψ', A, t[end]))

    # calculate the Chebyshev expansion of the scaled ω for the new N_min
    T = hcat(T, 2*ω_.*T[:, end] .- T[:, end-1])

    # calculate the χ for the new N_min
    sumval = sum(reduce(hcat, [g[n].*μ[n].*T[:, n] for n in 2:N_min]), dims=2)
    χ_next = prefactor .* (g[1] .* μ[1] .+ 2 .* sumval)

    # calculate error and the integral of χ
    error = maximum(abs.(χ_next .- χ))
    χ = χ_next
    sum_χ = (0.5 * (χ[1] + χ[end]) + sum(χ[2:end-1])) * Δω

    end_time = DateTime(now())
    Δt = end_time - start_time

    # output for error convergence and time measurement
    println("N = $N_min\t i = $i\t Error = $error\t sum_χ = $sum_χ\t Δsum = $(abs(sum_χ - sum_old))\t Δt = $Δt")

    # stopping criterion on N, best if all sites have same number of chebyshev expansion terms
    if N_min > N_max || error > 1 

      if N_min > N_max
        println("N_min > N_max")
      end
      if error > 1
        @error "Error > 1"
      end
      break
    end
    sum_old = sum_χ
  end
      
  return χ
end

function parse_arguments(E0::Float64, E1::Float64)
  if length(ARGS) == 0
    println("No command-line arguments provided. Reading in from stdin.")
    try 
      input_line = readline(stdin)
      input_args = split(input_line, ['"', '\''])
      tokens = split(input_args)
    catch e
      @error "Failed to read from stdin." exception=(e, catch_backtrace())
      exit(1)
    end
  else
    tokens = ARGS
  end

  if length(tokens) < 1
    println("Usage: julia Dynam_Corr.jl <J> [N_max] [cutoff]")
    exit(1)
  end

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
        exit(1)
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
        exit(1)
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

  J_input = input_args[1]
  
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
        exit(1)
      end
      # Convert arrays of arrays to a 2D Matrix
      J = reduce(vcat, J_matrix)
      J = reshape(J, row_lengths[1], length(J_matrix))
    catch e
      @error "Failed to parse J matrix: $e"
      exit(1)
    end
  else
    # Parse J as a scalar
    try
      J = parse(Float64, J_input)
    catch e
      @error "Failed to parse J as a Float64: $e"
      exit(1)
    end
  end

  N_max = abs(E0 + E1) < 600 ? abs(E0 + E1) : 600
  cutoff = 1e-8
  # Parse N_max or cutoff
  if length(input_args) == 2
    second_arg = input_args[2]
    if tryparse(Int, second_arg) !== nothing
      N_max = parse(Int, second_arg)
    elseif tryparse(Float64, second_arg) !== nothing
      cutoff = parse(Float64, second_arg)
    else
      @error "Second argument must be an integer (N_max) or a float (cutoff)."
      exit(1)
    end

  elseif length(input_args) == 3
    N_max = parse(Int, input_args[2])
    cutoff = parse(Float64, input_args[3])
  end

  return J, N_max, cutoff
end


function main()
# read in the Groundstate MPS
f = h5open("parent_calc_folder/psin_data.h5", "r")
ψ0 = read(f, "state_1/ψ", MPS)
close(f)
# show(ψ0)

N = length(ψ0)

# read in the Hamiltonian
f = h5open("parent_calc_folder/H_data.h5", "r")
H = read(f, "H", MPO)
close(f)

# read in the ground state energy
fr = open("parent_calc_folder/dmrg.out", "r")
lines = readlines(fr)
close(fr)

energy_line = findfirst(contains("List of E:"), lines) + 1
energy_array = eval(Meta.parse(lines[energy_line]))
E0 = energy_array[1]
E1 = energy_array[2]

# read in the operators
f = h5open("parent_calc_folder/operators_data.h5", "r")
sites = read(f, "sites", IndexSet)
I = Operators.Identity_op(sites)
Sz = [Operators.Szi_op(i, sites) for i in 1:N]
close(f)

# TODO: read in the J, to know where to stop with \omega

J , N_max, cutoff = parse_arguments(E0, E1)
println("N_max = $N_max")
println("cutoff = $cutoff")

len_ω = 1000
ω = collect(range(0.0001, stop=3, length=len_ω)) # if beginning with 0, use [2:end] and add 1 to len_ω
χ = zeros(length(Sz), length(ω))

# print the number of threads
println("Number of threads: $(nthreads())")
# main loop
@threads for i in eachindex(Sz)
  println("Calculating χ for Sz[$i]")
  χ[i, :] = Dynamic_correlator(ψ0, H, Sz[i], i, I, E0, E1, ω, N_max)
end

# write the results to a file for the plot
println(χ)

# TODO: add plotting here
end

main()

