# using ITensors, ITensorMPS
# let
#   # Create 100 spin-one indices
#   N = 100
#   sites = siteinds("S=1",N)

#   # Input operator terms which define
#   # a Hamiltonian matrix, and convert
#   # these terms to an MPO tensor network
#   # (here we make the 1D Heisenberg model)
#   os = OpSum()
#   for j=1:N-1
#     os += "Sz",j,"Sz",j+1
#     os += 0.5,"S+",j,"S-",j+1
#     os += 0.5,"S-",j,"S+",j+1
#   end
#   H = MPO(os,sites)

#   # Create an initial random matrix product state
#   psi0 = random_mps(sites)

#   # Plan to do 5 passes or 'sweeps' of DMRG,
#   # setting maximum MPS internal dimensions
#   # for each sweep and maximum truncation cutoff
#   # used when adapting internal dimensions:
#   nsweeps = 5
#   maxdim = [10,20,100,100,200]
#   cutoff = 1E-10

#   # Run the DMRG algorithm, returning energy
#   # (dominant eigenvalue) and optimized MPS
#   energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
#   println("Final energy = $energy")

#   nothing
# end

# # output

# # After sweep 1 energy=-137.954199761732 maxlinkdim=9 maxerr=2.43E-16 time=9.356
# # After sweep 2 energy=-138.935058943878 maxlinkdim=20 maxerr=4.97E-06 time=0.671
# # After sweep 3 energy=-138.940080155429 maxlinkdim=92 maxerr=1.00E-10 time=4.522
# # After sweep 4 energy=-138.940086009318 maxlinkdim=100 maxerr=1.05E-10 time=11.644
# # After sweep 5 energy=-138.940086058840 maxlinkdim=96 maxerr=1.00E-10 time=12.771
# # Final energy = -138.94008605883985



using ITensorMPS
using ITensorParallel
using ITensors

function heisenberg_2d(nx, ny)
  lattice = square_lattice(nx, ny; yperiodic=false)
  os = OpSum()
  for b in lattice
    os += 0.5, "S+", b.s1, "S-", b.s2
    os += 0.5, "S-", b.s1, "S+", b.s2
    os += "Sz", b.s1, "Sz", b.s2
  end
  return os
end

function heisenberg_2d_grouped(nx, ny)
  os_grouped_1 = OpSum[]
  os_grouped_2 = OpSum[]
  os_grouped_3 = OpSum[]

  # Horizontal terms
  for jy in 1:ny
    os_1 = OpSum()
    os_2 = OpSum()
    os_3 = OpSum()
    for jx in 1:(nx - 1)
      j1, j2 = (jx - 1) * ny + jy, jx * ny + jy
      os_1 += 0.5, "S+", j1, "S-", j2
      os_2 += 0.5, "S-", j1, "S+", j2
      os_3 += "Sz", j1, "Sz", j2
    end
    push!(os_grouped_1, os_1)
    push!(os_grouped_2, os_2)
    push!(os_grouped_3, os_3)
  end

  # Vertical terms
  os_1 = OpSum()
  os_2 = OpSum()
  os_3 = OpSum()
  for jx in 1:nx
    for jy in 1:(ny - 1)
      j1, j2 = (jx - 1) * ny + jy, (jx - 1) * ny + jy + 1
      os_1 += 0.5, "S+", j1, "S-", j2
      os_2 += 0.5, "S-", j1, "S+", j2
      os_3 += "Sz", j1, "Sz", j2
    end
  end
  push!(os_grouped_1, os_1)
  push!(os_grouped_2, os_2)
  push!(os_grouped_3, os_3)

  # Remove empty OpSums
  os_grouped_1 = filter(!isempty, os_grouped_1)
  os_grouped_2 = filter(!isempty, os_grouped_2)
  os_grouped_3 = filter(!isempty, os_grouped_3)
  return [os_grouped_1; os_grouped_2; os_grouped_3]
end

function main(nx, ny)
  os = heisenberg_2d(nx, ny)
  os_partition_manual = heisenberg_2d_grouped(nx, ny)
  os_partition_auto = partition(os; alg="chain_split")
  return (; os, os_partition_manual, os_partition_auto)
end