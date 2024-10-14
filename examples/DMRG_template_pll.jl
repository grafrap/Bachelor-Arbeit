# command for this script: mpiexecjl -n 4 julia DMRG_template_pll.jl 0.5 100 0.4 0 1 true true > output.txt 2> error.txt
###############################################################################
# packages #
###############################################################################
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

BLAS.set_num_threads(1)

# lib_dir = "."
# include(lib_dir*"/Operators.jl")
# include(lib_dir*"/AuxFunctions.jl")
# include(lib_dir*"/DMRGsweeps.jl")
# import .Operators
# import .AuxFunctions
# import .DMRGsweeps
# run this script with: julia DMRG_template.jl 0.5 10 23 38 1 0 true true
# functions #

function bcast(obj, root::Integer, comm::MPI.Comm)
  isroot = MPI.Comm_rank(comm) == root
  count = Ref{Clong}()
  if isroot
    buf = MPI.serialize(obj)
    count[] = length(buf)
  end
  MPI.Bcast!(count, root, comm)
  if !isroot
    buf = Array{UInt8}(undef, count[])
  end
  MPI.Bcast!(buf, root, comm)
  if !isroot
    obj = MPI.deserialize(buf)
  end
  return obj
end


function spinstate_sNSz(s,N,Sz;random="yes")
  #=
       function that creates a spin state from s,N,Sz

       Notes:
       - as of now, only s=1/2 and s=1 are possible
       - for s=1, the state created has minimal nr of "Z0"
       more info on the available spin states: https://itensor.github.io/ITensors.jl/dev/IncludedSiteTypes.html
       more on new s: https://github.com/ITensor/ITensors.jl/blob/main/src/lib/SiteTypes/src/sitetypes/spinhalf.jl
  =#
  # In total adds up to Sz, with available states from -s, -s+1, ..., s
    # TODO: Generalize this function for many s
    # general s
  if abs(Sz) > s*N
    throw(ArgumentError("Check s, N and Sz"))
  end

  state = fill("Dn", N)
  
  # s=1/2
  if s==1/2
    if N%2 == abs(2*Sz)%2
      for i in 1:Int(N/2+Sz)
        state[i] = "Up"
      end
      # example output ["Up", "Dn", "Up", "Up", "Up", "Dn", "Up", "Up", "Dn", "Dn"] for s = 0.5, Sz = 1, N = 10
    else
      throw(ArgumentError("Check s, N and Sz"))
    end
  # s=1
  elseif s==1
    if (2*Sz)%2 == 0
      if N%2==0
        for i in 1:Int(N/2+floor(Sz/2))
          state[i] = "Up"
        end
        if abs(Sz%2)==1
          state[Int(N/2+floor(Sz/2))+1] = "Z0"
        end
      elseif N%2==1
        for i in 1:Int(floor(N/2)+ceil(Sz/2))
          state[i] = "Up"
        end
        if abs(Sz%2)==0
          state[Int(floor(N/2)+ceil(Sz/2))+1] = "Z0"
        end
      end
      throw(ArgumentError("Check s, N and Sz"))
    end
    # example output: ["Z0", "Dn", "Up", "Up", "Dn", "Up", "Up", "Up", "Dn", "Dn"] for s = 1, Sz = 1, N = 10
  else
    println("ERROR: as of now, only s=1/2 and s=1 are possible")
    exit()
  end

  if random=="yes"
    shuffle!(state)
  end

  return(state)
end

function S2_op(Nsites, sites)
  #=
    S^2 operator

    S^2 = SpSm + Sz^2 - Sz
    SpSm = sum_{i,j} Sp(i)Sm(j)
    Sz^2 = sum_{i,j} Sz(i)Sz(j)
    Sz = sum_i Sz(i)
  =#
  
  ampo = AutoMPO()

  for i in 1:Nsites
    for j in 1:Nsites
      ampo += 1.,"S+",i,"S-",j
      ampo += 1.,"Sz",i,"Sz",j
    end
    ampo += -1.,"Sz",i
  end

  S2 = MPO(ampo, sites)

  return S2
end

function Szi_op(i, sites)
  #=
      Sz(i) operator
  =#

  ampo = AutoMPO()

  ampo += 1., "Sz", i

  Szi = MPO(ampo, sites)

  return Szi
end

function Hamiltonian(N::Int, sites; g::Vector = zeros(N), D::Vector = zeros(N),
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

  # Create the MPO
  # H = MPO(ampo, sites)

  return ampo
end


function DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi;maxdimi=300,
  maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
  #=
     DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz(i)
  =#

  # default sweeps
  sweeps0 = Sweeps(5)
  maxdim!(sweeps0, 20,50,100,100,200)
  cutoff!(sweeps0, 1E-5,1E-5,1E-5,1E-8,1E-8)

  if ψn===nothing 
    #ground state
    Ed,ψd = dmrg(H,ψi,sweeps0)
  else 
    #excited states
    Ed,ψd = dmrg(H,ψn,ψi,sweeps0;weight=w)
  end
  S2d = inner(ψd',S2,ψd)
  Szid = [inner(ψd',Szi[i],ψd) for i in eachindex(Szi)]

  # extra sweeps
  sweep1 = Sweeps(1)
  maxdim!(sweep1, maxdimi)
  cutoff!(sweep1, cutoff)

  j = 0
  E,ψ = Ed,ψd
  maxdim = maxdimi
  E_conv, S2_conv, Szi_conv = 0, 0, 0
  while true
    if ψn===nothing #ground state
      Ee,ψe = dmrg(H,ψd,sweep1)
    else #excited states
      Ee,ψe = dmrg(H,ψn,ψd,sweep1;weight=w)
    end
    S2e = inner(ψe',S2,ψe)
    Szie = [inner(ψe',Szi[i],ψe) for i in eachindex(Szi)]
    
    # Convergence checks, TODO: change to two times in a row converged or so
    if abs(Ed-Ee) < precE
      E_conv += 1
    else
      E_conv = 0
    end
    if E_conv == 2
      println("E converged")
    end

    if abs(S2d-S2e) < precS2
      S2_conv += 1
    else
      S2_conv = 0
    end
    if S2_conv == 2
      println("S² converged")
    end

    if maximum(abs.(Szid-Szie)) < precSzi
      Szi_conv += 1
    else
      Szi_conv = 0
    end
    if Szi_conv == 2 
      println("Sz(i) converged")
    end

    if abs(Ed-Ee) < precE && 
      maximum(abs.(Szid-Szie)) < precSzi
      j += 1
    else
      j = 0
    end

    if j==2
      E,ψ = Ee,ψe
      break
    end

    Ed,ψd = Ee,ψe
    S2d = S2e
    Szid = Szie

    maxdim += maxdimstep
    maxdim!(sweep1, maxdim)
  end

  return(E,ψ)
end


function write_simple_hdf5()
  # Open an HDF5 file for writing
  h5file = h5open("test.h5", "w")

  try
      # Create a group within the HDF5 file
      group = create_group(h5file, "my_group")

      # Write a simple dataset to the group
      write(group, "test_dataset", [1, 2, 3, 4, 5])
  catch e
      println("Error writing to HDF5 file: ", e)
  finally
      # Ensure the HDF5 file is closed
      close(h5file)
  end
end



# main #
function main()
start_time = DateTime(now())
BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()
ITensors.enable_threaded_blocksparse(true)

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
# Check if the correct number of arguments is provided
if rank == 0
  write_simple_hdf5()
  if length(ARGS) < 7
    println("Usage: julia DMRG_template.jl <s> <N> <J> <Sz> <nexc> <conserve_symmetry> <print_HDF5>")
    exit(1)
  end

  # physical parameters
  s = parse(Float64, ARGS[1])
  N = parse(Int, ARGS[2])
  J1 = parse(Float64, ARGS[3])
  
  # create the J matrix, with J1 on the off-diagonal TODO: generalize for all J matrices
  J = zeros(N,N)
  for i in 1:N-1
    J[i,i+1] = J1
    J[i+1,i] = J1
  end
  
  # other parameters
  Sz = parse(Float64, ARGS[4]) # Sz = 1/2 or 0
  @assert Sz == 1/2 || Sz == 0 "Sz must be 1/2 or 0"
  nexc = parse(Int, ARGS[5])
  
  conserve_symmetry, print_HDF5 = true, true
  try
    conserve_symmetry = parse(Bool, ARGS[6])
  catch e
    throw(ArgumentError("Failed to parse conserve_symmetry as Bool: $e"))
  end

  try
    print_HDF5 = parse(Bool, ARGS[7])
  catch e
    throw(ArgumentError("Failed to parse print_HDF5 as Bool: $e"))
  end


  
  # system
  Nsites = N
  if (2*s)%2 == 0
    sites = siteinds("S="*string(Int(s)),Nsites;conserve_sz=conserve_symmetry)
  else
    sites = siteinds("S="*string(Int(2*s))*"/2",Nsites;conserve_sz=conserve_symmetry)
  end
  
  # initial state
  statei = spinstate_sNSz(s, N, Sz)
  # println(statei)
  linkdim = 100 # variable to randomize initial MPS
  ψi = randomMPS(sites, statei, linkdim)
  # print(ψi)
  
  # S² operator
  S2 = S2_op(Nsites, sites)
  
  # Sz(i) operator
  Szi = [Szi_op(i, sites) for i in 1:Nsites]
  
  # Hamiltonian
  H = Hamiltonian(N, sites, J=J)
  Hpart = partition(H, nprocs)
else
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

sites = bcast(sites, 0, MPI.COMM_WORLD)
ψi = bcast(ψi, 0, MPI.COMM_WORLD)
S2 = bcast(S2, 0, MPI.COMM_WORLD)
Szi = bcast(Szi, 0, MPI.COMM_WORLD)
Sz = bcast(Sz, 0, MPI.COMM_WORLD)
H = bcast(H, 0, MPI.COMM_WORLD)
Hpart = bcast(Hpart, 0, MPI.COMM_WORLD)
nexc = bcast(nexc, 0, MPI.COMM_WORLD)
Nsites = bcast(Nsites, 0, MPI.COMM_WORLD)
print_HDF5 = bcast(print_HDF5, 0, MPI.COMM_WORLD)

precE = 1E-6
precS2 = 1E-5
precSzi = 1E-5
w = 1E5 # penalty for non-orthogonality

MPI.Barrier(MPI.COMM_WORLD)
mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
MPI.Barrier(MPI.COMM_WORLD)

# ground state
# E0, ψ0 = DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi)

# En = [E0]
# ψn = [ψ0]
ψn = [ψi]
En = []

# if nexc != 0 
#   H = MPO(mpo_sum_term, sites)
# end
# # excited states
# for i in 1:nexc
#   # TODO: This function is not callable in parallel context
#   E,ψ = DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi,ψn=ψn,w=w)
#   push!(En,E)
#   push!(ψn,ψ)
# end
MPI.Finalize()
if rank == 0
  # <ψn|S²|ψn>
  S2n = [inner(ψn[i]',S2,ψn[i]) for i in eachindex(ψn)]

  # <ψn|Sz(i)|ψn>
  Szin = [[inner(ψn[i]',Szi[j],ψn[i]) for j in 1:Nsites] for i in eachindex(ψn)]

  fw = open("Szi_Sz_pll=$(Sz).txt", "w")
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
  write(fw,"total time = "*
      string( Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now())
      - Dates.DateTime(start_time))) ) * "\n")

  close(fw)
println(ψn) # print MPS for external print to HDF5 file


# if rank == 0
# # Write the MPS data to the HDF5 file
# h5file = h5open("test.h5", "w")
# write(h5file, "test_dataset", [1, 2, 3, 4, 5])
# close(h5file)
#   if print_HDF5
#     println("Printing to HDF5 file")

#     # Open an HDF5 file for writing
#     h5file = h5open("psin_data.h5", "w")

#     # Write the MPS data to the HDF5 file
#     for (i, psi) in enumerate(ψn)
#       group = create_group(h5file, "state_$i")
#       write(group, "ψ", psi)
#     end

#     # Close the HDF5 file
#     close(h5file)
#   end
  
# end
end
end

main()
