module Operators

  using ITensors
  using ITensorMPS


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
  
  function Identity_op(sites)
    #=
      Identity operator
    =#
  
    I = MPO(sites, "Id")
  
    return I
  end
end