module Hamiltonian
using ITensors
using ITensorMPS
using ITensorParallel
function H(N::Int, sites; g::Vector = zeros(N), D::Vector = zeros(N),
  E::Vector = zeros(N), J::Matrix = zeros(N,N), Dij::Matrix = [zeros(3) for _ in 1:N, _ in 1:N], B::Vector = zeros(3))
  #= 
    General Hamiltonian for a spin chain with N sites
      H = sum_{i} g_i * μ_B * B * S_i (Zeeman term, the μ_B and B are included in g)
          + sum_{i} (D_i * (S^z_i)^2 + E_i * [(S^x_i)^2 - (S^y_i)^2]) (Anisotropy term)
          + sum_{i,j} J_{ij} * S_i * S_j (Heisenberg term)
          + sum_{i,j} D_{ij} * (S_i × S_j) (Dzyaloshinskii-Moriya term)
  =#

  @assert length(g) == N "g must have length N"
  @assert length(D) == N "D must have length N"
  @assert length(E) == N "E must have length N"
  @assert size(J) == (N,N) "J must have size (N,N)"
  @assert size(Dij) == (N,N) "Dij must have size (N,N)"

  ampo = OpSum()
  
  # create B * g, to get a vector of dimensions N x 3
  g = [[B[1]*g[i], B[2]*g[i], B[3]*g[i]] for i in 1:N]
  # Zeeman term (written in terms of S+, S-, Sz)
  for i in 1:N
    if g[i][1] != 0.0
      ampo += g[i][1], "Sx", i
    end 
    if g[i][2] != 0.0
      ampo += g[i][2], "Sy", i
    end
    if g[i][3] != 0.0
      ampo += g[i][3], "Sz", i
    end
  end


  # Anisotropy term (written in terms of S+, S-, Sz)
  for i in 1:N
    if D[i] != 0.0
      ampo += D[i], "Sz", i, "Sz", i
    end
    if E[i] != 0.0
      ampo += E[i], "Sx", i, "Sx", i 
      ampo -= E[i], "Sy", i, "Sy", i   
    end 
  end

  # Heisenberg coupling term
  for i in 1:N-1
    for j in i+1:N
      if J[i,j] != 0.0
        ampo += J[i,j], "Sz", i, "Sz", j
        ampo += J[i,j]/2, "S+", i, "S-", j
        ampo += J[i,j]/2, "S-", i, "S+", j
      end
    end
  end

  # Dzyaloshinskii-Moriya term
  for i in 1:N-1
    for j in i+1:N
      # Dij[i,j] = [Dij_x, Dij_y, Dij_z]
      if Dij[i,j][1] != 0.0
        ampo += Dij[i,j][1], "Sy", i, "Sz", j
        ampo -= Dij[i,j][1], "Sz", i, "Sy", j
      end
      if Dij[i,j][2] != 0.0
        ampo += Dij[i,j][2], "Sz", i, "Sx", j
        ampo -= Dij[i,j][2], "Sx", i, "Sz", j
      end
      if Dij[i,j][3] != 0.0
        ampo += Dij[i,j][3], "Sx", i, "Sy", j
        ampo -= Dij[i,j][3], "Sy", i, "Sx", j
      end      
    end
  end

  return ampo
end

end