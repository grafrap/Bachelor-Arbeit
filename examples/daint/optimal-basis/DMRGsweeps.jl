module DMRGsweeps

# packages #
using ITensors
###############################################################################

# DMRG sweeps (maxdim routine) until convergence in E and S^2
## GC, 03-05-2024
function DMRGmaxdim_convES2(H,ψi,precE,precS2,S2;maxdimi=300,maxdimstep=100,
    cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E and S^2
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
    S2d = inner(ψd',S2,ψd)

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

        if abs(Ed-Ee) < precE && abs(S2d-S2e) < precS2
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

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

# DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz(i)
## GC, 03-05-2024
function DMRGmaxdim_convES2Szi(H,ψi,precE,precS2,precSzi,S2,Szi;maxdimi=300,
    maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz(i)
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

# DMRG sweeps (maxdim routine) until convergence in E, S^2 and spin-string(d)
## GC, 03-05-2024
function DMRGmaxdim_convES2spinstrd(H,ψi,precE,precS2,precspinstrd,S2,
    func_spinstrd_ψ;maxdimi=300,maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E, S^2 and spin-string(d)
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
    S2d = inner(ψd',S2,ψd)
    spinstrdd = func_spinstrd_ψ(ψd)

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
        spinstrde = func_spinstrd_ψ(ψe)

        if abs(Ed-Ee) < precE && abs(S2d-S2e) < precS2 &&
            maximum(abs.(spinstrdd-spinstrde)) < precspinstrd
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
        spinstrdd = spinstrde

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

# DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz
## GC, 03-05-2024
function DMRGmaxdim_convES2Sz(H,ψi,precE,precS2,precSz,S2,Sz;maxdimi=300,
    maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E, S^2 and Sz
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
    S2d = inner(ψd,S2,ψd)
    Szd = inner(ψd,Sz,ψd)

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
        S2e = inner(ψe,S2,ψe)
        Sze = inner(ψe,Sz,ψe)

        if abs(Ed-Ee) < precE && abs(S2d-S2e) < precS2 && abs(Szd-Sze) < precSz
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
        Szd = Sze

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

# DMRG sweeps (maxdim routine) until convergence in E
## GC, 03-05-2024
function DMRGmaxdim_convE(H,ψi,precE;maxdimi=300,maxdimstep=100,cutoff=1E-8,
    ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E
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

        if abs(Ed-Ee) < precE
            j += 1
        else
            j = 0
        end

        if j==2
            E,ψ = Ee,ψe
            break
        end

        Ed,ψd = Ee,ψe

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

# DMRG sweeps (maxdim routine) until convergence in E, S^2 and spin-spin
## GC, 03-05-2024
function DMRGmaxdim_convES2spinspin(H,ψi,precE,precS2,precspinspin,S2,spinspin;
    maxdimi=300,maxdimstep=100,cutoff=1E-8,ψn=nothing,w=nothing)
    #=
       DMRG sweeps (maxdim routine) until convergence in E, S^2 and spin-spin
    =#

    ## default sweeps
    sweeps0 = Sweeps(5)
    maxdim!(sweeps0, 20,50,100,100,200)
    cutoff!(sweeps0, 1E-5,1E-5,1E-5,1E-8,1E-8)

    if ψn==nothing #ground state
        Ed,ψd = dmrg(H,ψi,sweeps0)
    else #excited states
        Ed,ψd = dmrg(H,ψn,ψi,sweeps0;weight=w)
    end
    S2d = inner(ψd',S2,ψd)

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

        if abs(Ed-Ee) < precE && abs(S2d-S2e) < precS2
			k = 0
	        for i in 1:length(spinspin)
	            spinspind = inner(ψd',spinspin[i],ψd)
	            spinspine = inner(ψe',spinspin[i],ψe)

	            if abs(spinspind-spinspine) > precspinspin
	                k = 1
	                break
	            end
	        end

			if k==0
	            j += 1
	        else
	            j = 0
	        end
        else
            j = 0
        end

        if j==2
            E,ψ = Ee,ψe
            break
        end

        Ed,ψd = Ee,ψe
        S2d = S2e

        maxdim += maxdimstep
        maxdim!(sweep1, maxdim)
    end

    return(E,ψ)
end

end
