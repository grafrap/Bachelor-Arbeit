using ITensors
using ITensorMPS
using HDF5
using InteractiveUtils
using LinearAlgebra
using Base.Threads

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
  t[1] = apply(B, ψ)
  t[2] = apply(H, t[1])

  for n in 3:N+1
    println("Calculating Chebyshev vector $n")
    t[n] = 2*apply(H, t[n-1]) - t[n-2]
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
  
function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, I::MPO, E0::Float64, E1::Float64, ω::Float64, N::Int)
  #=
    Implements the dynamical spin correlator for a single frequency
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#

  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  @assert ω < W "ω must be less than W"
  @assert ω > 0 "ω must be positive"

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  ω_ = ω/a - W_ # ω', scaled ω
  H_scaled = 1/a * (H - E0*I) - W_*I

  #calculate the Jackson dampening factors
  println("Calculating Jackson dampening factors")
  g = Jackson_dampening(N)

  # calculate the Chebyshev vectors
  println("Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N)

  # calculate the Chebyshev moments
  println("Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  println("Calculating Chebyshev expansion")
  T = Chebyshev_expansion(ω_, N)
  
  χ = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N+1]))
  return χ
end


function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, I::MPO, E0::Float64, E1::Float64, ω::Vector, N::Int)
  #=
    Implements the dynamical spin correlator for a vector of frequencies
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#

  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  for i in 1:length(ω)
    @assert ω[i] < W "ω must be less than W"
    @assert ω[i] > 0 "ω must be positive"
  end

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  H_scaled = 1/a * (H - E0*I) - W_*I

  #calculate the Jackson dampening factors
  println("Calculating Jackson dampening factors")
  g = Jackson_dampening(N)

  # calculate the Chebyshev vectors
  println("Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N)

  # calculate the Chebyshev moments
  println("Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  χ = zeros(length(ω))
  
  println("Calculating Chebyshev expansion")
  for j in eachindex(ω)
    ω_ = ω[j]/a - W_ # ω', scaled ω
    T = Chebyshev_expansion(ω_, N)
    χ[j] = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N+1]))
  end
  println(χ)
  return χ
end


function main()
# read in the Groundstate MPS
f = h5open("outputs/psin_data.h5", "r")
ψ0 = read(f, "state_1/ψ", MPS)
close(f)

N = length(ψ0)

# read in the Hamiltonian
f = h5open("outputs/H_data.h5", "r")
H = read(f, "H", MPO)
close(f)

# read in the ground state energy
fr = open("outputs/Szi_Sz_pll=0.0.txt", "r")
lines = readlines(fr)
close(fr)

energy_line = findfirst(contains("List of E:"), lines) + 1
energy_array = eval(Meta.parse(lines[energy_line]))
E0 = energy_array[1]
E1 = energy_array[2]

# read in the Sz(i)
f = h5open("outputs/operators_data.h5", "r")
# S2 = read(f, "S2", MPO)
I = read(f, "I", MPO)
Sz::Vector{MPO} = []
group = f["Sz"]
for i in 1:N
  push!(Sz, read(group, "Sz_$i", MPO))
end
close(f)

# calculate the dynamical correlator
len_ω = 41
ω = collect(range(0, 2, len_ω))[2:end]
i = 1
N = 15
χ = zeros(length(Sz), len_ω-1)
# TODO: test with results from the paper: formula 9 and fig 5
# Take J = 1, J2 = 0.19J, ΔJ = 0.03J, N_sites = 18, S = 1, Sz = 1
# mpiexecjl -n 4 julia DMRG_template_pll_Energyextrema.jl 1 18 1.0 0 0 true true > outputs/output.txt 2> outputs/error.txt
@threads for i in eachindex(Sz)
  println("Calculating χ for Sz[$i]")
  χ[i, :] = Dynamic_corrolator(ψ0, H, Sz[i], I, E0, E1, ω, N)
end
# χ = Dynamic_corrolator(ψ0, H, Sz[i], I, E0, E1, ω, N)
# println("χ(ω = $ω) = $χ")
println(χ)
end

main()