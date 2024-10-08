# packages #
using MPI: MPI
MPI.Init()

using ITensors
using ITensorMPS
using ITensorParallel
using Dates
using Random
using HDF5
using LinearAlgebra: BLAS
using Strided
# parallel dmrg in C++: https://github.com/emstoudenmire/parallelDMRG/blob/master/parallel_dmrg.h

###############################################################################
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

# functions #
function spinstate_sNSz(s,N,Sz;random="yes")
    #=
       function that creates a spin state from s,N,Sz

       Notes:
       - as of now, only s=1/2 and s=1 are possible
       - for s=1, the state created has minimal nr of "Z0"
    =#

    # s=1/2
    if s==1/2
        if N%2 == abs(2*Sz)%2 && abs(Sz) <= s*N
            state = ["Dn" for i in 1:N]
            for i in 1:Int(N/2+Sz)
                state[i] = "Up"
            end
            if random=="yes"
                shuffle!(state)
            end
        else
            println("ERROR: check s, N and Sz")
            exit()
        end
    # s=1
    elseif s==1
        if (2*Sz)%2 == 0 && abs(Sz) <= s*N
            state = ["Dn" for i in 1:N]
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
            if random=="yes"
                shuffle!(state)
            end
        else
            println("ERROR: check s, N and Sz")
            exit()
        end
    else
        println("ERROR: as of now, only s=1/2 and s=1 are possible")
        exit()
    end

    return(state)
end

function S2_op(Nsites,sites)
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

    S2 = MPO(ampo,sites)

    return(S2)
end

function Szi_op(i,sites)
    #=
		Sz(i) operator
	=#

    ampo = AutoMPO()

    ampo += 1.,"Sz",i

    Szi = MPO(ampo,sites)

    return(Szi)
end

function H_Heis_J1J2_1D_OBC(N,J1,J2,sites)
    #=
        Hamiltonian for open-ended J1-J2 Heisenberg chain

        (1)--(2)...(3)

        --, J1*vec{S}.vec{S}
        ..., J2*vec{S}.vec{S}
    =#

    ampo = OpSum()

    #J1
    if J1 != 0.0
        for i in 1:2:N-1
            ampo += J1,"Sz",i,"Sz",i+1
            ampo += J1/2,"S+",i,"S-",i+1
            ampo += J1/2,"S-",i,"S+",i+1
        end
    end

    #J2
    if J2 != 0.0
        for i in 2:2:N-1
            ampo += J2,"Sz",i,"Sz",i+1
            ampo += J2/2,"S+",i,"S-",i+1
            ampo += J2/2,"S-",i,"S+",i+1
        end
    end

    # H = OpSum(ampo)

    return ampo

end

function DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi;maxdimi=300,
    maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz(i)
    =#
    if !(H isa MPO)
      H = MPO(H, siteinds(ψi))
    end
    # default sweeps

    sweeps0 = Sweeps(5)
    maxdim!(sweeps0, 20,50,100,100,200)
    cutoff!(sweeps0, 1E-5,1E-5,1E-5,1E-8,1E-8)

    if ψn==nothing #ground state
        Ed,ψd = dmrg(H,ψi,sweeps0)
    else #excited states
        Ed,ψd = dmrg(H,ψn,ψi,sweeps0;weight=w)
    end
    S2d = inner(ψd',S2,ψd)
    Szid = [inner(ψd',Szi[i],ψd) for i in 1:length(Szi)]

    # extra sweeps
    sweep1 = Sweeps(1)
    maxdim!(sweep1, maxdimi)
    cutoff!(sweep1, cutoff)

    j = 0
    E,ψ = Ed,ψd
    maxdim = maxdimi
    while true
        if ψn==nothing #ground state
            Ee,ψe = dmrg(H,ψd,sweep1)
        else #excited states
            Ee,ψe = dmrg(H,ψn,ψd,sweep1;weight=w)
        end
        S2e = inner(ψe',S2,ψe)
        Szie = [inner(ψe',Szi[i],ψe) for i in 1:length(Szi)]

        if abs(Ed-Ee) < precE && abs(S2d-S2e) < precS2 &&
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
###############################################################################
function prt(number)
    println("number = ", number)
end

# main #
let
start_time = DateTime(now())
BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()
@show Threads.nthreads()

ITensors.enable_threaded_blocksparse(true)
@show ITensors.using_threaded_blocksparse()
# physical parameters
s = 1/2
N = 10
J1 = 23
J2 = 38

prt(227)
# other parameters
Sz = +1
nexc = 0
precE = 0.00001
precS2 = 0.00001
precSzi = 0.00001
w = 10000.0 #penalty for non-orthogonality
linkdim = 100 #variable to randomize initial MPS

# open files
fw = open("Szi_Sz=$(Sz)_pll.txt", "w")

ITensors.enable_threaded_blocksparse(true)
# system
Nsites = N
if (2*s)%2 == 0
    sites = siteinds("S="*string(Int(s)),Nsites;conserve_sz=true)
  else
    sites = siteinds("S="*string(Int(2*s))*"/2",Nsites;conserve_sz=true)
end
prt(248)
# initial state
statei = spinstate_sNSz(s,N,Sz)

prt(254)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)
if rank == 0
    ψi = randomMPS(sites,statei,linkdim)
    S2 = S2_op(Nsites,sites)
    Szi = [Szi_op(i,sites) for i in 1:Nsites]
    H = H_Heis_J1J2_1D_OBC(N,J1,J2,sites)
    ℋs = partition(H, nprocs)
else
    ψi = nothing
    S2 = nothing
    Szi = nothing
    H = nothing
    ℋs = nothing
end
# ψi = randomMPS(sites,statei,linkdim)

# S^2 operator

# Sz(i) operator
  
prt(262)
sites = bcast(sites, 0, MPI.COMM_WORLD)
ψi = bcast(ψi, 0, MPI.COMM_WORLD)
S2 = bcast(S2, 0, MPI.COMM_WORLD)
Szi = bcast(Szi, 0, MPI.COMM_WORLD)
H = bcast(H, 0, MPI.COMM_WORLD)
ℋs = bcast(ℋs, 0, MPI.COMM_WORLD)
prt(267)
# Hamiltonian

# in_partition_alg = "sum_split"
# println("rank = ", rank, " ℋs:", ℋs)
which_proc = MPI.Comm_rank(MPI.COMM_WORLD) + 1
MPI.Barrier(MPI.COMM_WORLD)
@show ℋs
@show which_proc
@show ℋs[which_proc] 
mpo_sum_term = MPISumTerm(MPO(ℋs[which_proc], sites), MPI.COMM_WORLD)
# println("rank = ", rank, " sum_term:", mpo_sum_term)
prt(273)
# Hs = [MPO(ℋ, sites) for ℋ in ℋs]
# H = DistributedSum(Hs)
# H_ = DistributedSum(Hs)
# run DMRG
## groundstate
MPI.Barrier(MPI.COMM_WORLD)
E0,ψ0 =  DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi)

## excited states
En = [E0]
ψn = [ψ0]
for n in 1:nexc
  Eaux,ψaux = DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi,ψn=ψn,w=w)
  println()
  push!(En,Eaux)
  push!(ψn,ψaux)
end

prt(291)
# println(ψn)
# <psin|S^2|psin>
S2n = [inner(ψn[n]',S2,ψn[n]) for n in 1:length(En)]

# <psin|Sz(i)|psin>
Szin = [[0. for i in 1:Nsites] for n in 1:length(En)]
for n in 1:length(En)
    for i in 1:Nsites
        Szin[n][i] = inner(ψn[n]',Szi[i],ψn[n])
    end
end
prt(302)
# outputs
write(fw, "List of E:\n")
write(fw, string(En), "\n")
write(fw, "\n")
write(fw, "List of S^2:\n")
write(fw, string(S2n), "\n")
write(fw, "\n")
write(fw, "List of Szi:\n")
write(fw, string(Szin), "\n")
write(fw, "----------\n\n")
write(fw,"total time = "*
    string( Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now())
    - Dates.DateTime(start_time))) ) * "\n")

# close file
close(fw)

println("Printing to HDF5 file")

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
###############################################################################
