module DMRGSweeps
using ITensors
using ITensorMPS
using ITensorParallel
using MPI

# function to check if two AbstractMPS have common site indices
  function check_hascommoninds(::typeof(siteinds), A::AbstractMPS, B::AbstractMPS)
    # check if A and B have the same length
    N = length(A)
    if length(B) ≠ N
      throw(
        DimensionMismatch(
          "$(typeof(A)) and $(typeof(B)) have mismatched lengths $N and $(length(B))."
        ),
      )
    end
    # check if A and B have common site indices
    for n in 1:N
      !hascommoninds(siteinds(A, n), siteinds(B, n)) && error(
        "$(typeof(A)) A and $(typeof(B)) B must share site indices. On site $n, A has site indices $(siteinds(A, n)) while B has site indices $(siteinds(B, n)).",
      )
    end
    return nothing
  end

  # parallel DMRG function for excited states
  function dmrg(H::MPISumTerm{ProjMPO}, Ms::Vector{MPS}, psi0::MPS, sweeps::Sweeps; weight = true, kwargs...)
    #=
      DMRG for excited states
    =#

    # check if H, Ms and psi0 have common site indices
    check_hascommoninds(siteinds, H.term.H, psi0)
    check_hascommoninds(siteinds, H.term.H, psi0')
    for M in Ms
      check_hascommoninds(siteinds, M, psi0)
    end

    # permute the indices of H and Ms
    H = permute(H.term.H, (linkind, siteinds, linkind))
    Ms = permute.(Ms, Ref((linkind, siteinds, linkind)))

    # check if weight is reasonable
    if weight <= 0
      error(
        "weight parameter should be > 0.0 in call to excited-state dmrg (value passed was weight=$weight)",
      )
    end

    # calcualte the new Hamiltonian, with the projector. This will ensure that the ground state is orthogonal to the excited states
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
    
    if ψn===nothing 
      #ground state
      Ed,ψd = ITensorMPS.dmrg(H,ψi,sweeps0)
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
          println("E converged")
        end
      end

      if abs(S2d-S2e) < precS2
        S2_conv += 1
      else
        S2_conv = 0
      end
      if S2_conv == 2
        if rank == 0
          println("S² converged")
        end
      end

      if maximum(abs.(Szid-Szie)) < precSzi || maximum(abs.(Szie)) < 1e-3
        Szi_conv += 1
      else
        Szi_conv = 0
      end
      if Szi_conv == 2 
        if rank == 0
          println("Sz(i) converged")
        end
      end
      if rank == 0
        println("maximum(abs.(Szid-Szie)) = ",maximum(abs.(Szid-Szie)))
        println("maximum(abs.(Szie)) = ", maximum(abs.(Szie)))
      end
        # check if all converged and stop if so
      if abs(Ed-Ee) < precE && 
        (maximum(abs.(Szid-Szie)) < precSzi || maximum(abs.(Szie)) < 1e-3)
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
end
