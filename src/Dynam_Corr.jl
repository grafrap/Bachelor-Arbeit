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
  N_max = Int(round(W))
  if N_max < 500
    N_max = 500
  end
  if N != 500
    N_max = N
  end
  println(stderr, "N_max = $N_max")

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
    println(stderr, "N = $N_min\t i = $i\t Error = $error\t sum_χ = $sum_χ\t Δsum = $(abs(sum_χ - sum_old))\t Δt = $Δt")

    # stopping criterion on N, best if all sites have same number of chebyshev expansion terms
    if N_min > N_max || error > 1 

      if N_min > N_max
        println(stderr, "N_min > N_max")
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

N_max = 0
# calculate the dynamical correlator
if length(ARGS) >= 1
  N_max = ARGS[1]
elseif -E0 - E1 < 600
  N_max = 600
else
  N_max = -E0 - E1
end

len_ω = 1000
ω = collect(range(0.0001, stop=3, length=len_ω)) # if beginning with 0, use [2:end] and add 1 to len_ω
χ = zeros(length(Sz), length(ω))

# main loop
@threads for i in eachindex(Sz)
  println(stderr, "Calculating χ for Sz[$i]")
  χ[i, :] = Dynamic_correlator(ψ0, H, Sz[i], i, I, E0, E1, ω, N_max)
end

# write the results to a file for the plot
println(χ)

# TODO: add plotting here
end

main()

