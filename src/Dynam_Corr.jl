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
import .Operators

function krylov_elements(H::MPO, ψ::MPS, N::Int)
  #=
    Implements the Krylov elements K_n = Ĥ^(n-1)*|ψ⟩
    Orthogonalizes the Krylov elements using the Gram-Schmidt process
    and return the reduced Hamiltonian H_red (n x n matrix) 
  =#

  K = Vector{MPS}(undef, N)
  K[1] = ψ
  # normalize the first element
  normalize!(K[1])
  H_red = zeros(N, N)

  for n in 2:N
    K[n] = apply(H, K[n-1])
    for i in 1:(n-1)
      # orthonormalize via Gram-Schmidt
      K[n] -= inner(K[i], K[n]) * K[i]
    end
    normalize!(K[n])
  end

  # calculate the reduced Hamiltonian
  for i in 1:N
    for j in 1:N
      H_red[i, j] = inner(K[i]', H, K[j])
    end
  end

  return H_red
end

function energy_truncation(H::MPO, N::Int, ϵ::Float64, t::MPS)

  println(stderr, "Calculating the energy truncation")

  for i in eachindex(t)
    sites = siteinds(t[1])

    # Build the Krylov subspace
    K = Vector{MPS}(undef, N)
    K[1] = t
    normalize!(K[1])
    for n in 2:N
      K[n] = apply(H, K[n-1])
      for i in 1:(n-1)
        # Orthogonalize via Gram-Schmidt
        K[n] -= inner(K[i], K[n]) * K[i]
      end
      normalize!(K[n])
    end

    # Calculate the reduced Hamiltonian
    H_red = zeros(N, N)
    for i in 1:N
      for j in 1:N
        H_red[i, j] = inner(K[i]', H, K[j])
      end
    end

    # Diagonalize H_red
    λ, v = eigen(H_red)
    println(stderr, "Eigenvalues: ", λ)

    # Truncate the states
    P = Operators.Identity_op(sites)

    # Truncate the states by subtracting |v_i⟩⟨v_i|
    count = 0
    for i in 1:N
      if λ[i] > ϵ
        count += 1
        # Construct |v_i⟩ as an MPS
        for i in 1:N
          P -= outer(v[:, i], v[:, i])
        end
      end
    end

    t = apply(P, t)
    
    if count == 0
      println(stderr, "No states to truncate")
    else
      println(stderr, "Truncated $count states")
    end

  end
end

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
  t[1] = apply(B, ψ; cutoff=1e-6)
  energy_truncation(H, 30, 1.0, t[1])
  t[2] = apply(H, t[1]; cutoff=1e-6)
  energy_truncation(H, 30, 1.0, t[2])


  for n in 3:N+1
    println(stderr, "Calculating Chebyshev vector $n")
    t[n] = 2*apply(H, t[n-1]; cutoff=1e-8)
    t[n] = add(t[n], -t[n-2]; cutoff=1e-8)
    energy_truncation(H, 30, 1.0, t[n])
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
  
function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, i::Int, I::MPO, E0::Float64, E1::Float64, ω::Float64, N::Int)
  #=
    Implements the dynamical spin correlator for a single frequency
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#

  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  println(stderr, "W = $W")
  @assert ω < W "ω must be less than W"
  @assert ω > 0 "ω must be positive"

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  ω_ = ω/a - W_ # ω', scaled ω
  H_scaled = 1/a * (H - E0*I) - W_*I
  # H_pows = build_H_pow(H_scaled, 30)

  
  #calculate the Jackson dampening factors
  println(stderr, "Calculating Jackson dampening factors")
  g = Jackson_dampening(N)

  # calculate the Chebyshev vectors
  println(stderr, "Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N)
  
  # calculate the Chebyshev moments
  println(stderr, "Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)
  
  # calculate the Chebyshev expansion of the scaled ω
  println(stderr, "Calculating Chebyshev expansion")
  T = Chebyshev_expansion(ω_, N)
  
  χ = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N]))
  
  while true
    N += 1
    g = Jackson_dampening(N)
    t_next = 2*apply(H_scaled, t[end]) - t[end-1]
    # energy_truncation(H_scaled, 30, 1.0, t_next)
    
    push!(t, t_next)
    push!(μ, inner(ψ', A, t[end]))
    push!(T, 2*ω_*T[end] - T[end-1])
    
    χ_next = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N]))
    error = abs(χ_next - χ)
    println(stderr, "N = $N, i = $i, Error = $error")
    if error < 1e-4 
      break
    end
    χ = χ_next
  end
  return χ
end


function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, i::Int, I::MPO, E0::Float64, E1::Float64, ω::Vector; N_min::Int=3, abstol::Float64=1e-5)
  #=
  Implements the dynamical spin correlator for a vector of frequencies
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#
  if N_min < 3
    N_min = 3
  end
  # TODO: stopping criterion for the Chebyshev expansion based on W or J or precision to sum to s*(s+1)
  N_max = 1000

  ϵ = 0.025
  # W = 4 * ω[end] # W* = -E0 + E1
  W = 2 * ω[end]#-E0 - E1
  for i in 1:length(ω)
    @assert ω[i] < W "ω must be less than W"
    @assert ω[i] > 0 "ω must be positive"
  end

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  H_scaled = 1/a * (H - E0*I) - W_*I
  # H_pows = build_H_pow(H_scaled, 30)

  #calculate the Jackson dampening factors
  println(stderr, "Calculating Jackson dampening factors")
  g = Jackson_dampening(N_min)

  # calculate the Chebyshev vectors
  println(stderr, "Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N_min)

  # calculate the Chebyshev moments
  println(stderr, "Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  χ = zeros(length(ω))
  χ_next = zeros(length(ω))
  
  println(stderr, "Calculating Chebyshev expansion")
  ω_ = ω./a .- W_ # ω', scaled ω
  Δω = ω[2] - ω[1]
  sum_old = 0
  T = Chebyshev_expansion(ω_, N_min)

  prefactor = 1 ./(a * π * sqrt.(1 .- ω_.^2))
  sumval = sum(reduce(hcat, [g[n].*μ[n].*T[:, n] for n in 2:N_min]), dims=2)
  χ = prefactor .* (g[1] * μ[1] .+ 2 .* sumval)
  bond_array = zeros(3)

  while true 
    start_time = DateTime(now())

    N_min += 1
    g = Jackson_dampening(N_min)

    bond_array[1] = maxlinkdim(t[end])
    t_next = 2*apply(H_scaled, t[end]; cutoff=1e-8)
    bond_array[2] = maxlinkdim(t_next)
    t_next = add(t_next, -t[end-1]; cutoff=1e-8)
    bond_array[3] = maxlinkdim(t_next)
    println(stderr, "Bond dimensions: ", bond_array)    
    energy_truncation(H_scaled, 30, 1.0, t_next)

    push!(t, t_next)

    push!(μ, inner(ψ', A, t[end]))

    T = hcat(T, 2*ω_.*T[:, end] .- T[:, end-1])

    sumval = sum(reduce(hcat, [g[n].*μ[n].*T[:, n] for n in 2:N_min]), dims=2)

    χ_next = prefactor .* (g[1] .* μ[1] .+ 2 .* sumval)

    error = maximum(abs.(χ_next .- χ))
    χ = χ_next
    sum_χ = (0.5 * (χ[1] + χ[end]) + sum(χ[2:end-1])) * Δω

    end_time = DateTime(now())
    Δt = end_time - start_time

    println(stderr, "N = $N_min\t i = $i\t Error = $error\t sum_χ = $sum_χ\t Δsum = $(abs(sum_χ - sum_old))\t Δt = $Δt")

    if error < abstol || N_min > N_max || error > 1 #|| abs(sum_χ - sum_old) < abstol
      # print what of the stopping criteria was met
      if error < abstol
        println(stderr, "Error < abstol")
      end
      if N_min > N_max
        println(stderr, "N_min > N_max")
      end
      if error > 1
        @error "Error > 1"
      end
      if abs(sum_χ - sum_old) < 2e-5
        println(stderr, "Δsum < 2e-5")
      end
      break
    end
    sum_old = sum_χ
  end
      
  return χ
end


function main()
# read in the Groundstate MPS
f = h5open("outputs/psin_data.h5", "r")
ψ0 = read(f, "state_1/ψ", MPS)
close(f)
# show(ψ0)

N = length(ψ0)

# read in the Hamiltonian
f = h5open("outputs/H_data.h5", "r")
H = read(f, "H", MPO)
close(f)

# read in the ground state energy
fr = open("outputs/Szi_Sz_pll=1.0.txt", "r")
lines = readlines(fr)
close(fr)

energy_line = findfirst(contains("List of E:"), lines) + 1
energy_array = eval(Meta.parse(lines[energy_line]))
E0 = energy_array[1]
E1 = energy_array[2]

# read in the operators
f = h5open("outputs/operators_data.h5", "r")
sites = read(f, "sites", IndexSet)
I = Operators.Identity_op(sites)
Sz = [Operators.Szi_op(i, sites) for i in 1:N]



# calculate the dynamical correlator
len_ω = 1000
W = -E0 - E1
ω = collect(range(0.005, stop=2, length=len_ω)) # if beginning with 0, use [2:end] and add 1 to len_ω
println("length of ψ: ",length(ψ0))
i = 1
N_min = 5
χ = zeros(length(Sz), length(ω))
# TODO: test with results from the paper: formula 9 and fig 5
# Take J = 1, J2 = 0.19J, ΔJ = 0.03J, N_sites = 18, S = 1, Sz = 1
# mpiexecjl -n 4 julia DMRG_template_pll_Energyextrema.jl 1 18 1.0 0 0 true true > outputs/output.txt 2> outputs/error.txt
@threads for i in eachindex(Sz)
  println(stderr, "Calculating χ for Sz[$i]")
  χ[i, :] = Dynamic_corrolator(ψ0, H, Sz[i], i, I, E0, E1, ω; N_min)
end

println(χ)
end

main()

