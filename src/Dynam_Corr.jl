using ITensors
using ITensorMPS
using HDF5
using InteractiveUtils
using LinearAlgebra
using Base.Threads
using Dates
using Plots

lib_dir = "."
include(lib_dir*"/Operators.jl")
include(lib_dir*"/Customspace.jl")
import .Operators


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
    time = DateTime(now())
    t[n] = 2*apply(H, t[n-1]; cutoff=1e-8)
    t[n] = add(t[n], -t[n-2]; cutoff=1e-8)
    end_time = DateTime(now())
    println("Chebyshev_vectors, N=$n\t, Δt = $(end_time - time)")
  end

  return t
end

function Chebyshev_moments(ψ::MPS, A::MPO, t::Array{MPS, 1}, i::Int)
  #=
    Implements the Chebyshev moments μ_n = ⟨ψ|Â*T_n(Ĥ')*B^|ψ⟩ = ⟨ψ|Â∣t_n⟩ (first half)
    and μ_{n+n'} = 2*<t_n|t_{n'}> - μ_{n-n'} (second half)
  =#
  μ = zeros(2*length(t)-1)

  for n in 1:length(t)
    time = DateTime(now())
    μ[n] = inner(ψ', A, t[n])
    end_time = DateTime(now())
    println("N=$n\t i=$i\t Δt = $(end_time - time)")
  end

  n = length(t)
  for n_ in 1:length(t)-1
    time = DateTime(now())
    μ[n+n_] = 2*inner(t[n],t[n_+1]) - μ[n-n_]
    end_time = DateTime(now())
    println("N=$(n+n_)\t i=$i\t Δt = $(end_time - time)")
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
  for idx in 1:length(ω)
    @assert ω[idx] < -E0 - E1 "ω must be less than W"
    @assert ω[idx] > 0 "ω must be positive"
  end
  
  # parameters
  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)

  H_scaled = add(1/a * add(H, -E0*I; cutoff=cutoff), -W_*I; cutoff=cutoff)

  ω_ = ω./a .- W_ # ω', scaled ω
  Δω = ω[2] - ω[1] # frequency step for integration

  #calculate the Jackson dampening factors
  g = Jackson_dampening(N_max)

  # calculate the Chebyshev vectors
  t = Chebyshev_vectors(A, H_scaled, ψ, Int(ceil(N_max/2.)))
  print("length of t = $(length(t))\n")

  # calculate the Chebyshev moments
  μ = Chebyshev_moments(ψ, A, t, i)

  # calculate the Chebyshev expansion of the scaled ω
  χ = zeros(length(ω))
  
  # calculate the Chebyshev expansion of the scaled ω
  T = Chebyshev_expansion(ω_, N_max)

  # calculate the χ for the initial N_min
  prefactor = 1 ./(a * π * sqrt.(1 .- ω_.^2))
  sumval = sum(reduce(hcat, [g[n].*μ[n].*T[:, n] for n in 2:N_max]), dims=2)
  χ = prefactor .* (g[1] * μ[1] .+ 2 .* sumval)

  # calculate error and the integral of χ
  integral_χ = (0.5 * (χ[1] + χ[end]) + sum(χ[2:end-1])) * Δω
  println("N = $N_max\t i = $i\t sum_χ = $integral_χ")
  return χ
end

function parse_arguments(E0::Float64, E1::Float64)
  input_args = []
  if length(ARGS) == 0
    println("No command-line arguments provided. Reading in from stdin.")
    try 
      input_line = readline(stdin)
      input_line = split(input_line, ['"', '\''])
      tokens = split(input_line[1])
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

  N_max = Int(round(abs(E0 + E1) > 600 ? abs(E0 + E1) : 600))
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

# read in the ground state energy
fr = open("parent_calc_folder/dmrg.out", "r")
lines = readlines(fr)
close(fr)

energy_line = findfirst(contains("List of E:"), lines) + 1
energy_array = eval(Meta.parse(lines[energy_line]))
E0 = energy_array[1]
E1 = energy_array[end]

J , N_max, cutoff = parse_arguments(E0, E1)
f = h5open("parent_calc_folder/psin_data.h5", "r")
ψ0 = read(f, "state_1/ψ", MPS)
close(f)
# show(ψ0)

N = length(ψ0)

# read in the Hamiltonian
f = h5open("parent_calc_folder/H_data.h5", "r")
H = read(f, "H", MPO)
close(f)
# read in the operators
f = h5open("parent_calc_folder/operators_data.h5", "r")
sites = read(f, "sites", IndexSet)
I = Operators.Identity_op(sites)
Sz = [Operators.Szi_op(i, sites) for i in 1:N]
close(f)

J_mean = 0.0
if typeof(J) == Matrix{Float64}
  # assign the avg of the nonzero elements to J_mean
  J_mean = sum(J) / length(J[J .!= 0])
else
  J_mean = J
end
  
println("N_max = $N_max")
println("cutoff = $cutoff")
println("J = $J_mean")

len_ω = 500 * max(1, Int(round(abs(J_mean))))
ω = collect(range(0.0001, stop=2*abs(J_mean), length=len_ω)) # if beginning with 0, use [2:end] and add 1 to len_ω
χ = zeros(length(Sz), length(ω))

# print the number of threads
println("Number of threads: $(nthreads())")
# main loop
@threads for i in eachindex(Sz)
  println("Calculating χ for Sz($i)")
  χ[i, :] = Dynamic_correlator(ψ0, H, Sz[i], i, I, E0, E1, ω, N_max)
end

# write the results to a file for the plot
println(χ)

# write χ to a seperate out file
f = open("parent_calc_folder/chi_data.txt", "w")
write(f, string(χ), "\n")

min_val = minimum(χ)
χ = χ .- min_val
max_val = maximum(χ)
χ = χ ./ max_val

χ_transposed = permutedims(χ)

# plot the histogram for χ
tickpoints = range(0, len_ω, length=11)
heatmap(χ_transposed, xlabel="Sites", ylabel="Frequencies [J]", title="2D Histogram of Matrix for N = $N_max", size=(800, 600), margin=10Plots.mm)
yticks!(tickpoints, string.([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]))
savefig("histogram.png")
end

main()

