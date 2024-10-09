module AuxFunctions

# packages #
using ITensors
using Random
###############################################################################

# function that creates a spin state from s,N,Sz
## GC, 13-12-2022
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

# function that creates a fermion state from Nsites,Ne,Sz
## GC, 13-12-2022
function fermionstate_NsitesNeSz(Nsites,Ne,Sz;random="yes")
    #=
       function that creates a fermion state from Nsites,Ne,Sz
    =#

    Ne_up,Ne_dn = 0,0
    if Ne%2 == 0 && (2*Sz)%2 == 0
		Ne_up = Int(Ne÷2 + Sz)
		Ne_dn = Int(Ne÷2 - Sz)
	elseif Ne%2 == 1 && abs(2*Sz)%2 == 1
		Ne_up = Int(Ne÷2 + (Sz+1/2))
		Ne_dn = Int(Ne÷2 - (Sz-1/2))
    else
        println("ERROR: check Ne and Sz")
        exit()
    end

    state = ["Emp" for i in 1:Nsites]
    if Ne <= Nsites
        for i in 1:Ne_up
            state[i] = "Up"
        end
        for i in (Ne_up+1):Ne
            state[i] = "Dn"
        end
    elseif Ne > Nsites
        for i in 1:(Ne-Nsites)
            state[i] = "UpDn"
        end
        for i in (Ne-Nsites+1):Ne_up
            state[i] = "Up"
        end
        for i in Ne_up+1:Nsites
            state[i] = "Dn"
        end
    end
    if random=="yes"
        shuffle!(state)
    end

    return(state)
end

# function that calculates spin-string(d)
## GC, 14-12-2022
function calc_spinstrd(ψ,Szi_bulk,Rzi_bulk)
    #=
        function that calculates spin-string(d)

        spin-string(d) = <ψ| Sz(k) Rz(k+1) Rz(k+2) ... Rz(k+d-1) Sz(k+d) |ψ>
        where:
        Rz(k) = Exp[I*pi*Sz(k)]
        k, k+1, ..., k+d are bulk sites
	=#

    spinstrd = [0.0+0.0im for d in 1:length(Szi_bulk)-1]
    ψbra = ψ
    for d in 1:length(Szi_bulk)-1
        ψket = apply(Szi_bulk[1+d],ψ)
        for i in d:-1:2
            ψket = apply(Rzi_bulk[i],ψket)
        end
        ψket = apply(Szi_bulk[1],ψket)

        spinstrd[d] = inner(ψbra,ψket)
    end

    return(spinstrd)
end

# function that finds states with total spin S
## GC, 29-12-2022
function find_Sstates(S,En,ψn,S2n;precS=0.1)
	#=
		function that finds states with total spin S
	=#

	EnS,ψnS,index_nS = [],[],[]
	for n in 1:length(En)
		if abs(S2n[n]-S*(S+1)) < precS
			push!(EnS,En[n])
			push!(ψnS,ψn[n])
            push!(index_nS,n)
        end
    end

	return(EnS,ψnS,index_nS)
end

# function to calculate bare spin spectral weights (and excitation energies)
# 	between GSs with total spin SGS and states with total spin S
## GC, 29-12-2022
function calc_ssw0(SGS,EnGS,ψnGS,S,EnS,ψnS,Sziop,Sxiop,ISyiop,precE)
	#=
		function to calculate bare spin spectral weights (and excitation
			energies) between GSs with total spin SGS and states with total
			spin S

		ssw0(i)_nGS_nS = sum_{mGS,mS} [ |<nGS,mGS|Sx(i)|nS,mS>|^2
			+ |<nGS,mGS|Sy(i)|nS,mS>|^2 + |<nGS,mGS|Sz(i)|nS,mS>|^2 ]
	=#

	# consistency checks
	for nGS in 1:length(EnGS)÷Int(2*SGS+1)
		for mGS in 2:Int(2*SGS+1)
			if abs( EnGS[(nGS-1)*Int(2*SGS+1)+1] -
                EnGS[(nGS-1)*Int(2*SGS+1)+mGS] ) > 100*precE
                println("ERROR: non-degenerate states in GS multiplet nr " *
                    string(nGS))
                exit()
			end
        end
    end
	if length(EnS)%Int(2*S+1) != 0
		println("WARNING: inconsistent nr of states")
		println("TRY: proceed without last multiplet")
		ψnS = ψnS[1:length(ψnS)-length(ψnS)%Int(2*S+1)]
    end
	for nS in 1:length(EnS)÷Int(2*S+1)
		for mS in 2:Int(2*S+1)
			if abs(EnS[(nS-1)*Int(2*S+1)+1]-EnS[(nS-1)*Int(2*S+1)+mS]) >
                100*precE
				println("ERROR: non-degenerate states in multiplet nr " *
                    string(nS))
				exit()
            end
        end
    end

	excE_nGS_nS = [0.0 for nGS in 1:length(EnGS)÷Int(2*SGS+1),
        nS in 1:length(EnS)÷Int(2*S+1)]
	ssw0i_nGS_nS = [0.0 for i in 1:length(Sziop),
		nGS in 1:length(EnGS)÷Int(2*SGS+1),
        nS in 1:length(EnS)÷Int(2*S+1)]
    for nGS in 1:length(EnGS)÷Int(2*SGS+1)
		for nS in 1:length(EnS)÷Int(2*S+1)
			# excitation energies
			excE_nGS_nS[nGS,nS] = EnS[(nS-1)*Int(2*S+1)+1] -
                EnGS[(nGS-1)*Int(2*SGS+1)+1]

			# bare spin spectral weights
			for i in 1:length(Sziop)
				for mGS in 1:Int(2*SGS+1)
					for mS in 1:Int(2*S+1)
						#|<GS|Sz(i)|psinS>|^2
						ssw0i_nGS_nS[i,nGS,nS] += abs(
							inner(ψnGS[(nGS-1)*Int(2*SGS+1)+mGS],Sziop[i],
                            ψnS[(nS-1)*Int(2*S+1)+mS]) )^2
						#|<GS|Sx(i)|psinS>|^2
						ssw0i_nGS_nS[i,nGS,nS] += abs(
							inner(ψnGS[(nGS-1)*Int(2*SGS+1)+mGS],Sxiop[i],
                            ψnS[(nS-1)*Int(2*S+1)+mS]) )^2
						#|<GS|Sy(i)|psinS>|^2 = |<GS|ISy(i)|psinS>|^2
						ssw0i_nGS_nS[i,nGS,nS] += abs(
							inner(ψnGS[(nGS-1)*Int(2*SGS+1)+mGS],ISyiop[i],
                            ψnS[(nS-1)*Int(2*S+1)+mS]) )^2
                    end
                end
            end
        end
    end

	return(excE_nGS_nS,ssw0i_nGS_nS)
end

# function to calculate bare spin spectral weights (and excitation energies)
# 	between GSs and excited states, with known degeneracies
## GC, 02-01-2023
function calc_ssw0_alt(degGS,EnGS,ψnGS,degn,En,ψn,Sziop,Sxiop,ISyiop)
	#=
		function to calculate bare spin spectral weights (and excitation
			energies) between GSs and excited states, with known degeneracies

        ssw0(i)_nGS_n = sum_{mGS,m} [ |<nGS,mGS|Sx(i)|n,m>|^2
    		+ |<nGS,mGS|Sy(i)|n,m>|^2 + |<nGS,mGS|Sz(i)|n,m>|^2 ]
	=#

    excE_nGS_n = [0.0 for nGS in 1:length(degGS), n in 1:length(degn)]
	ssw0i_nGS_n = [0.0 for i in 1:length(Sziop), nGS in 1:length(degGS),
        n in 1:length(degn)]
    for nGS in 1:length(degGS)
		for n in 1:length(degn)
			# excitation energies
			excE_nGS_n[nGS,n] = En[sum(degn[1:n])] - EnGS[sum(degGS[1:nGS])]

			# bare spin spectral weights
			for i in 1:length(Sziop)
				for mGS in 1:degGS[nGS]
					for m in 1:degn[n]
						#|<GS|Sz(i)|psin>|^2
						ssw0i_nGS_n[i,nGS,n] += abs(
							inner(ψnGS[sum(degGS[1:nGS])-degGS[nGS]+mGS],
                            Sziop[i],ψn[sum(degn[1:n])-degn[n]+m]) )^2
						#|<GS|Sx(i)|psin>|^2
                        ssw0i_nGS_n[i,nGS,n] += abs(
                            inner(ψnGS[sum(degGS[1:nGS])-degGS[nGS]+mGS],
                            Sxiop[i],ψn[sum(degn[1:n])-degn[n]+m]) )^2
						#|<GS|Sy(i)|psin>|^2 = |<GS|ISy(i)|psin>|^2
                        ssw0i_nGS_n[i,nGS,n] += abs(
                            inner(ψnGS[sum(degGS[1:nGS])-degGS[nGS]+mGS],
                            ISyiop[i],ψn[sum(degn[1:n])-degn[n]+m]) )^2
                    end
                end
            end
        end
    end

	return(excE_nGS_n,ssw0i_nGS_n)
end

# function to calculate degeneracies
## GC, 02-01-2023
function calc_deg(En,precE)
    #=
        function to calculate degeneracies

        Note: elements of last multiplet are excluded since it is not possible
            to check if there are missing states
    =#
    degn = []
    nref, count = 1, 1
    for naux in 2:length(En)
        if abs(En[naux] - En[nref]) < 100*precE
            count += 1
        else
            push!(degn,count)
            nref = naux
            count = 1
        end
    end

    return degn
end

end
