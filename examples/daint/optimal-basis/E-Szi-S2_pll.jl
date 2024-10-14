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
using Dates
using LinearAlgebra: BLAS
using Strided

BLAS.set_num_threads(1)

###############################################################################
# my libraries #
###############################################################################
lib_dir = "."
include(lib_dir*"/Operators.jl")
include(lib_dir*"/AuxFunctions.jl")
include(lib_dir*"/DMRGsweeps.jl")
import .Operators
import .AuxFunctions
import .DMRGsweeps
###############################################################################

# distances in units of dCC
# angles in degrees

###############################################################################
# my functions #
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

function Hamiltonian(t1,tau1,ta,tb,NN_list,U,sites;V1=0.0,pS2=0.0)
    ampo = OpSum()

    N = length(sites)

    ## t1
    if t1 != 0.0
        for i in 1:N, j in NN_list[i]
            if (i,j)!=(11,14) && (i,j)!=(14,11)
                ampo += -t1,"Cdagup",i,"Cup",j #up
                ampo += -t1,"Cdagdn",i,"Cdn",j #dn
            end
        end
    end

    ## tau1
    if tau1 != 0.0
        for (i,j) in [(11,14)]
            ampo += tau1,"Cdagup",i,"Cup",j #up
            ampo += tau1,"Cdagdn",i,"Cdn",j #dn
            #hc
            ampo += tau1,"Cdagup",j,"Cup",i
            ampo += tau1,"Cdagdn",j,"Cdn",i
        end
    end

    ## ta
    if ta != 0.0
        for (i,j) in [(12,15),(13,16)]
            ampo += ta,"Cdagup",i,"Cup",j #up
            ampo += ta,"Cdagdn",i,"Cdn",j #dn
            #hc
            ampo += ta,"Cdagup",j,"Cup",i
            ampo += ta,"Cdagdn",j,"Cdn",i
        end
    end

    ## tb
    if tb != 0.0
        for (i,j) in [(12,16),(13,15)]
            ampo += tb,"Cdagup",i,"Cup",j #up
            ampo += tb,"Cdagdn",i,"Cdn",j #dn
            #hc
            ampo += tb,"Cdagup",j,"Cup",i
            ampo += tb,"Cdagdn",j,"Cdn",i
        end
    end

    ## U
	if U != 0.0
	    for i in 1:N
	        ampo += U,"Nupdn",i
	    end
	end

    ## V1
	if V1 != 0.0
        for i in 1:N, j in NN_list[i]
            ampo += 0.5*V1,"Ntot",i,"Ntot",j
        end
	end

    # p*S^2
    if pS2 != 0.0
    	for i in 1:N
            for j in 1:N
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    # H = MPO(ampo,sites)

    return ampo
end

function da(theta)
    theta_rad = deg2rad(theta) 
    return sqrt(5.5-1.5*cos(theta_rad))
end

function db(theta)
    theta_rad = deg2rad(theta) 
    return sqrt(5.5+1.5*cos(theta_rad))
end

function tpi(r,t1,qpi)
    return -t1*exp(qpi*(1-r))
end

function tsigma(r,qpi,tsigma0)
    qsigma = qpi
    return tsigma0*exp(qsigma*(1-r))
end

function ta(theta,t1,qpi,tsigma0)
    theta_rad = deg2rad(theta)
    da_val = da(theta)
    tpi_val = tpi(da_val,t1,qpi)
    tsigma_val = tsigma(da_val,qpi,tsigma0)
    s = tsigma_val*3/4*sin(theta_rad)^2/(5.5 - 1.5*cos(theta_rad))
    p = tpi_val*(cos(theta_rad) - 3/4*sin(theta_rad)^2/(5.5 - 1.5*cos(theta_rad)))
    return s + p
end

function tb(theta,t1,qpi,tsigma0)
    theta_rad = deg2rad(theta)
    db_val = db(theta)
    tpi_val = tpi(db_val,t1,qpi)
    tsigma_val = tsigma(db_val,qpi,tsigma0)
    s = -tsigma_val*3/4*sin(theta_rad)^2/(5.5 + 1.5*cos(theta_rad))
    p = tpi_val*(cos(theta_rad) + 3/4*sin(theta_rad)^2/(5.5 + 1.5*cos(theta_rad)))
    return s + p
end

function tau1(theta,t1)
    theta_rad = deg2rad(theta)
    return -t1*cos(theta_rad)
end

###############################################################################
# main #
###############################################################################
let
start_time = DateTime(now())
BLAS.set_num_threads(1)
NDTensors.Strided.disable_threads()
ITensors.enable_threaded_blocksparse(true)

# list of first neighbors
NN_list = [[3, 4], [4, 5], [1, 6], [1, 2, 7], [2, 8], [3, 9], [4, 9, 10], 
    [5, 10], [6, 7, 12], [7, 8, 13], [12, 13, 14], [9, 11], [10, 11], 
    [11, 15, 16], [14, 17], [14, 18], [15, 19, 20], [16, 20, 21], [17, 22], 
    [17, 18, 23], [18, 24], [19, 25], [20, 25, 26], [21, 26], [22, 23], 
    [23, 24]]

# physical parameters
N = length(NN_list)
theta = 0 #degree
t1 = 2.7*1000 #meV
qpi = 1.646
tsigma0 = 7.276*1000 #meV
U_over_t1 = 2.7
U = abs(t1)*U_over_t1
V1_over_U = 0.4
V1 = U*V1_over_U

# other parameters
Ne = N
Sz = parse(Int64,ARGS[1])
nexc = 0
precE = 0.001
precSzi = 0.001
precS2 = 0.001
linkdim = 100 #variable to randomize initial MPS
w = 10000.0 #penalty for non-orthogonality
cutoff=1E-8
maxdimstep=500

rank = MPI.Comm_rank(MPI.COMM_WORLD)
nprocs = MPI.Comm_size(MPI.COMM_WORLD)


# system
if rank == 0
  Nsites = N
  sites = siteinds("Electron",Nsites;conserve_nf=true,conserve_sz=true)
  
  # initial state
  statei = AuxFunctions.fermionstate_NsitesNeSz(N,Ne,Sz)
  ψi = randomMPS(sites,statei,linkdim)
  
  # Hamiltonian
  ta_val = ta(theta,t1,qpi,tsigma0)
  tb_val = tb(theta,t1,qpi,tsigma0)
  tau1_val = tau1(theta,t1)
  H = Hamiltonian(t1,tau1_val,ta_val,tb_val,NN_list,U,sites,V1=V1)
  
  # Sz(i) operator
  Szi = [Operators.Szi_op(i,sites) for i in 1:N]
  
  # S^2 operator
  S2 = Operators.S2_op(Nsites,sites)
  Hpart = partition(H, nprocs)
else
  sites = nothing
  ψi = nothing
  S2 = nothing
  Szi = nothing
  H = nothing
  Hpart = nothing
end

sites = bcast(sites, 0, MPI.COMM_WORLD)
ψi = bcast(ψi, 0, MPI.COMM_WORLD)
S2 = bcast(S2, 0, MPI.COMM_WORLD)
Szi = bcast(Szi, 0, MPI.COMM_WORLD)
H = bcast(H, 0, MPI.COMM_WORLD)
Hpart = bcast(Hpart, 0, MPI.COMM_WORLD)

MPI.Barrier(MPI.COMM_WORLD)
mpo_sum_term = MPISumTerm(MPO(Hpart[rank+1], sites), MPI.COMM_WORLD)
MPI.Barrier(MPI.COMM_WORLD)
# run DMRG
## groundstate
E0,ψ0 = DMRGsweeps.DMRGmaxdim_convES2Szi(mpo_sum_term,ψi,precE,precS2,precSzi,
S2,Szi,cutoff=cutoff,maxdimstep=maxdimstep)
println()
## excited states
En,ψn = [E0],[ψ0]
for n in 1:nexc
  println()
end

# <psin|S^2|psin>
S2n = [inner(ψn[n]',S2,ψn[n]) for n in 1:length(En)] 

# <psin|Sz(i)|psin>
Szin = [[0. for i in 1:N] for n in 1:length(En)]
for n in 1:length(En)
  for i in 1:N
    Szin[n][i] = inner(ψn[n]',Szi[i],ψn[n])
  end
end
MPI.Finalize()
# open files
fw = open("Sz_$(Sz)_pll.txt", "w")
# outputs
write(fw, "#List of E:\n")
write(fw, string(En), "\n")
write(fw, "\n")
write(fw, "#List of S^2:\n")
write(fw, string(S2n), "\n")
write(fw, "\n")
write(fw, "#List of Sz(i):\n")
write(fw, string(Szin), "\n")
write(fw, "----------\n\n")
write(fw,"total time = "*
    string( Dates.canonicalize(Dates.CompoundPeriod(Dates.DateTime(now())
    - Dates.DateTime(start_time))) ) * "\n")

# close files
close(fw)


end
