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
using InteractiveUtils


BLAS.set_num_threads(1)

# lib_dir = "."
# include(lib_dir*"/Operators.jl")
# include(lib_dir*"/AuxFunctions.jl")
# include(lib_dir*"/DMRGsweeps.jl")
# import .Operators
# import .AuxFunctions
# import .DMRGsweeps
# run this script with: mpiexecjl -n 4 julia DMRG_template_pll.jl 0.5 10 0.4 0 1 true true > output.txt

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
    println(stderr, "ERROR: as of now, only s=1/2 and s=1 are possible")
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
  # println(stderr, ampo)

  return ampo
end

# function hascommoninds(::typeof(siteinds), A::AbstractMPS, B::AbstractMPS)
#   N = length(A)
#   for n in 1:N
#     !hascommoninds(siteinds(A, n), siteinds(B, n)) && return false
#   end
#   return true
# end

function check_hascommoninds(::typeof(siteinds), A::AbstractMPS, B::AbstractMPS)
  N = length(A)
  if length(B) ≠ N
    throw(
      DimensionMismatch(
        "$(typeof(A)) and $(typeof(B)) have mismatched lengths $N and $(length(B))."
      ),
    )
  end
  for n in 1:N
    !hascommoninds(siteinds(A, n), siteinds(B, n)) && error(
      "$(typeof(A)) A and $(typeof(B)) B must share site indices. On site $n, A has site indices $(siteinds(A, n)) while B has site indices $(siteinds(B, n)).",
    )
  end
  return nothing
end

# parallel DMRG function for excited states
function dmrg(H::MPISumTerm{ProjMPO}, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight = true, kwargs...)
  check_hascommoninds(siteinds, H.term.H, psi0)
  check_hascommoninds(siteinds, H.term.H, psi0')
  for M in Ms
    check_hascommoninds(siteinds, M, psi0)
  end
  H = permute(H.term.H, (linkind, siteinds, linkind))
  Ms = permute.(Ms, Ref((linkind, siteinds, linkind)))
  if weight <= 0
    error(
      "weight parameter should be > 0.0 in call to excited-state dmrg (value passed was weight=$weight)",
    )
  end
  PMM = ITensorMPS.ProjMPO_MPS(H, Ms; weight)
  return ITensorMPS.dmrg(MPISumTerm(PMM, MPI.COMM_WORLD), psi0, sweeps; kwargs...)
end


function DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi;maxdimi=300,
  maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
  #=
     DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz(i)
  =#
  rank = MPI.Comm_rank(MPI.COMM_WORLD)
  # default sweeps
  sweeps0 = Sweeps(5)
  maxdim!(sweeps0, 20,50,100,100,200)
  cutoff!(sweeps0, 1E-5,1E-5,1E-5,1E-8,1E-8)
  
  # debug prints
  # if rank == 0
  #   println(stderr, "\n\nH:")
  #   println(stderr, typeof(H))
  #   println(stderr, H)
  #   println(stderr, "\n\n")
  #   println(stderr, H.term)
  #   println(stderr, "\n\n")
  # end

  if ψn===nothing 
    #ground state
    # if rank == 0
    #   println("Method signature: ", @which ITensorMPS.dmrg(H, ψi, sweeps0))
    # end
    Ed,ψd = ITensorMPS.dmrg(H,ψi,sweeps0)
  else 
    #excited states
    # if rank == 0
    #   println("Method signature: ", @which dmrg(H, ψn, ψi, sweeps0; weight=w))
    # end
    Ed,ψd = dmrg(H,ψn,ψi,sweeps0;weight=w)
  end
  S2d = inner(ψd',S2,ψd)
  Szid = [inner(ψd',Szi[i],ψd) for i in eachindex(Szi)]

  # extra sweeps
  sweep1 = Sweeps(1)
  maxdim!(sweep1, maxdimi)
  cutoff!(sweep1, cutoff)

  # variables for loops
  j = 0
  E,ψ = Ed,ψd
  maxdim = maxdimi

  # variables for convergence
  E_conv, S2_conv, Szi_conv = 0, 0, 0
  while true

    if ψn===nothing 
      #ground state
      Ee,ψe = ITensorMPS.dmrg(H,ψd,sweep1)
    else
      #excited states
      Ee,ψe = dmrg(H,ψn,ψd,sweep1;weight=w)
    end

    # create the inner products
    S2e = inner(ψe',S2,ψe)
    Szie = [inner(ψe',Szi[i],ψe) for i in eachindex(Szi)]
    
    # check convergence
    if abs(Ed-Ee) < precE
      E_conv += 1
    else
      E_conv = 0
    end
    if E_conv == 2
      if rank == 0
        println(stderr, "E converged")
      end
    end

    if abs(S2d-S2e) < precS2
      S2_conv += 1
    else
      S2_conv = 0
    end
    if S2_conv == 2
      if rank == 0
        println(stderr, "S² converged")
      end
    end

    if maximum(abs.(Szid-Szie)) < precSzi
      Szi_conv += 1
    else
      Szi_conv = 0
    end
    if Szi_conv == 2 
      if rank == 0
        println(stderr, "Sz(i) converged")
      end
    end

    # check if all converged and stop if so
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

    # update variables
    Ed,ψd = Ee,ψe
    S2d = S2e
    Szid = Szie

    maxdim += maxdimstep
    maxdim!(sweep1, maxdim)
  end

  return E,ψ
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

  # Check if the correct number of arguments is provided
  if length(ARGS) < 7
    println(stderr, "Usage: julia DMRG_template.jl <s> <N> <J> <Sz> <nexc> <conserve_symmetry> <print_HDF5>")
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
  Sz = parse(Float64, ARGS[4]) # Sz = 1/2 or 0 for now
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
  linkdim = 100 # variable to randomize initial MPS
  ψi = randomMPS(sites, statei, linkdim)
  
  # S² operator
  S2 = S2_op(Nsites, sites)
  
  # Sz(i) operator
  Szi = [Szi_op(i, sites) for i in 1:Nsites]
  
  # Hamiltonian
  H = Hamiltonian(N, sites, J=J)

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

# DMRG precision parameters
precE = 1E-6
precS2 = 1E-5
precSzi = 1E-5
w = 1E5 # penalty for non-orthogonality

# sum term for parallel DMRG
MPI.Barrier(MPI.COMM_WORLD)
mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
MPI.Barrier(MPI.COMM_WORLD)

# copy of the initial mpo
mpo_sum_term_copy = mpo_sum_term

# ground state
E0, ψ0 = DMRGmaxdim_convES2Szi(mpo_sum_term, ψi, precE, precS2, precSzi, S2, Szi)

# save ground state
En = [E0]
ψn = [ψ0]

# if rank == 0
#   # println(stderr, "mpo_sum_term")
#   # println(stderr, mpo_sum_term)
#   println(stderr, "mpo_sum_term_copy")
#   println(stderr, mpo_sum_term_copy)
# end

# excited states
for _ in 1:nexc
  MPI.Barrier(MPI.COMM_WORLD)
  mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
  MPI.Barrier(MPI.COMM_WORLD)
  E,ψ = DMRGmaxdim_convES2Szi(mpo_sum_term,ψi,precE,precS2,precSzi,S2,Szi,ψn=ψn,w=w)
  push!(En,E)
  push!(ψn,ψ)
end

if rank == 0
  # <ψn|S²|ψn>
  S2n = [inner(ψn[i]',S2,ψn[i]) for i in eachindex(ψn)]
  
  # <ψn|Sz(i)|ψn>
  Szin = [[inner(ψn[i]',Szi[j],ψn[i]) for j in 1:Nsites] for i in eachindex(ψn)]
  
  time = Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now()) - Dates.DateTime(start_time)))

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
  write(fw,"total time = "* string(time) * "\n")
  close(fw)

  @show time
  
  if print_HDF5
  
    println(stderr, "Printing to HDF5 file")

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
