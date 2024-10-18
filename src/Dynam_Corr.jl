using ITensors
using ITensorMPS
using HDF5
using InteractiveUtils
using LinearAlgebra

function chebyshev_coefficients(H::Matrix, order::Int)
  #=
    Calculate the Chebyshev coefficients of a matrix H
  =#
  # Initialize the Chebyshev coefficients
  N = size(H, 1)
  T0 = I(N)
  T1 = H

  # Initialize the Chebyshev coefficients
  coefficients = zeros(ComplexF64, N)

  # Calculate the Chebyshev coefficients
  coefficients[1] = trace(T0) / N
  coefficients[2] = trace(T1) / N

  for i in 3:order
    T2 = 2 * H * T1 - T0
    coefficients[i] = trace(T2) / N
    T0 = T1
    T1 = T2
  end
  return coefficients
end

function Dynamic_corrolator(ψ::MPS, H::MPO, Sz::Array{MPO,1},  I::MPO, E0::Float64, ω, i)
  #=
    Implements the dynamical spin correlator
    χ(ω) = <ψ|S^z_i δ(ωI - Ĥ - E_0) S^z_i|ψ>
    where δ is the Dirac delta function
  =#

  N = length(ψ)
  χ = zeros(ComplexF64, N)

  # calculate δ(ωI - Ĥ - E_0)
  H_bar = (ω + E0) * I - H 

  


  # Scale and shift the Hamiltonian to the interval [-1, 1]

  # Calculate the Chebyshev coefficients
  order = 50
  coefficients = chebyshev_coefficients(delta_H_bar, order)

  χ[i] = 0.0
  println("χ(ω = $ω) = $χ")

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

# read in the Sz(i)
f = h5open("outputs/operators_data.h5", "r")
S2 = read(f, "S2", MPO)
I = read(f, "I", MPO)
Sz::Vector{MPO} = []
group = f["Sz"]
for i in 1:N
  push!(Sz, read(group, "Sz_$i", MPO))
end
close(f)

# calculate the dynamical correlator
ω = 0.1
i = 1
χ = Dynamic_corrolator(ψ0, H, Sz, I, E0, ω, i)
println("χ(ω = $ω) = $χ")

end

main()