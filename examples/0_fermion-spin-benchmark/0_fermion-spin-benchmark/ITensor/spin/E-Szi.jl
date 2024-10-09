###############################################################################
# packages #
###############################################################################
using ITensors
using Dates
using Random
using LinearAlgebra

BLAS.set_num_threads(1)

###############################################################################
# my functions #
###############################################################################
function spinstate_sNSz(s,N,Sz;random="yes")
    #=
       function that creates a spin state from s,N,Sz

       Notes:
       - as of now, only s=1/2 and s=1 are possible
       - for s=1, the state created has minimal nr of "Z0"

       GC, 13-12-2022
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

function Szi_op(i,sites)
    #=
		Sz(i) operator

        GC, 17-01-2022
	=#

    ampo = AutoMPO()

    ampo += 1.,"Sz",i

    Szi = MPO(ampo,sites)

    return(Szi)
end

function H_Heis_J1J2_1D_OBC(N,J1,J2,sites;pS2=0.0,hSz=0.0)
    #=
        Hamiltonian for open-ended J1-J2 Heisenberg chain

        (1)--(2)...(3)

        --, J1*vec{S}.vec{S}
        ..., J2*vec{S}.vec{S}

        Notes:
        - if J1=J2, Heisenberg chain is recovered
        - pS2: adds term pS2*S^2 to Hamiltonian
        - hSz: adds term hSz*Sz to Hamiltonian

        GC, 13-12-2022
    =#

    ampo = AutoMPO()

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

    # pS2
    if pS2 != 0.0
    	for i in 1:N
            for j in 1:N
                ampo += pS2,"S+",i,"S-",j
                ampo += pS2,"Sz",i,"Sz",j
            end
            ampo += -pS2,"Sz",i
        end
    end

    # hSz
    if hSz != 0.0
    	for i in 1:N
            ampo += hSz,"Sz",i
        end
    end

    H = MPO(ampo,sites)

    return H
end

function DMRGmaxdim_convESzi(H,ψi,precE,precSzi,Szi;maxdimi=300,
    maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E and Sz(i)

       GC, 09-10-2024
    =#

    # default sweeps
    sweeps0 = Sweeps(5)
    maxdim!(sweeps0, 20,50,100,100,200)
    cutoff!(sweeps0, 1E-5,1E-5,1E-5,1E-8,1E-8)

    if ψn==nothing #ground state
        Ed,ψd = dmrg(H,ψi,sweeps0)
    else #excited states
        Ed,ψd = dmrg(H,ψn,ψi,sweeps0;weight=w)
    end
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
        Szie = [inner(ψe',Szi[i],ψe) for i in 1:length(Szi)]

        if abs(Ed-Ee) < precE && maximum(abs.(Szid-Szie)) < precSzi
            j += 1
        else
            j = 0
        end

        if j==2
            E,ψ = Ee,ψe
            break
        end

        Ed,ψd = Ee,ψe
        Szid = Szie

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

###############################################################################
# main #
###############################################################################
let
start_time = DateTime(now())

# physical parameters
s = 1/2
N = parse(Int64,ARGS[1])
J = 4/10

# other parameters
if (2*s)%2==0 || N%2==0
    Sz = 0
else
    Sz = 1/2
end
nexc = 1
precE = 0.00001
precSzi = 0.00001
w = 10000.0 #penalty for non-orthogonality
linkdim = 100 #variable to randomize initial MPS

# open files
fw = open("N_$(N).txt", "w")

# system
if (2*s)%2 == 0
    sites = siteinds("S="*string(Int(s)),N;conserve_sz=true)
else
    sites = siteinds("S="*string(Int(2*s))*"/2",N;conserve_sz=true)
end

# initial state
statei = spinstate_sNSz(s,N,Sz)
ψi = randomMPS(sites,statei,linkdim)

# Hamiltonian
H = H_Heis_J1J2_1D_OBC(N,J,J,sites)

# Sz(i) operator
Szi = [Szi_op(i,sites) for i in 1:N]

# run DMRG
## groundstate
E0,ψ0 = DMRGmaxdim_convESzi(H,ψi,precE,precSzi,Szi)
println()
## excited states
En = [E0]
ψn = [ψ0]
for n in 1:nexc
    Eaux,ψaux = DMRGmaxdim_convESzi(H,ψi,precE,precSzi,Szi,ψn=ψn,w=w)
    println()
    push!(En,Eaux)
    push!(ψn,ψaux)
end

# <psin|Sz(i)|psin>
Szin = [[0. for i in 1:N] for n in 1:length(En)]
for n in 1:length(En)
    for i in 1:N
        Szin[n][i] = inner(ψn[n]',Szi[i],ψn[n])
    end
end

# outputs
write(fw, "#List of E:\n")
write(fw, string(En), "\n")
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
