# src/Customspace.jl
using ITensors, ITensorMPS
using LinearAlgebra

export SpinSiteType

struct SpinSiteType
  #=
    Generic type for a spin site with a given S
  =#
  S::Rational{Int} 
end

tag(s::SpinSiteType) = s.S


function generate_spin_states(S::Rational{Int})
  #=
    Returns an array of the spin states for a given S.
  =#
  states = []
  for m in -S:S
    if m == 0
      push!(states, "Z0")
    elseif m > 0
      push!(states, "Up$(m)")
    else
      push!(states, "Dn$(-m)")
    end
  end
  return states
end

function state_to_index(S::Rational{Int})
  #=
    Returns a dictionary mapping the state names to their corresponding index values for a given S.
  =#
  states = generate_spin_states(S)
  state_index_map = Dict{String, Int}()
  for (i, state) in enumerate(states)
    state_index_map[state] = i
  end
  return state_index_map
end

function state_to_index(s::Index)
  #=
    Returns a dictionary mapping the state names to their corresponding index values for a given Index s.
  =#
  index_str = string(s)
  S = match(r"S=(\d+//\d+)", index_str).captures[1]
  state_index_map = state_to_index(parse(Rational{Int}, S))
  return state_index_map
end

function ITensors.state(s::SpinSiteType, n::Int)
  #=
    Return an IndexVal corresponding to the state n for the SpinSiteType s.
  =#
  return IndexVal(n, siteind(s))
end

function ITensors.state(s::SpinSiteType, str::String)
  #=
    Return an IndexVal corresponding to the state named str for the SpinSiteType s.
  =#
  IndexVal(state_to_index(s.S)[str], siteind(s))
end

function ITensors.state(s::Index, str::String)
  #=
    Return an ITensor corresponding to the state named str for the Index s. The returned ITensor will have s as its only index.
  =#
  state_index_map = state_to_index(s)
  arr = zeros(Int, length(state_index_map))
  arr[state_index_map[str]] = 1
  return arr
end

function ITensors.siteind(s::SpinSiteType; addtags="", kwargs...)
  #=
    Returns the site index for a given site type
  =#
  sp = space(s; kwargs...)
  return Index(sp, "Site,S=$(tag(s)), $addtags")
end

function ITensors.siteind(s::SpinSiteType, n::Integer; addtags="", kwargs...)
  #=
    Returns the site index for a given site type and site number
  =#
  sp = space(s; kwargs...)
  return Index(sp, "S=$(tag(s)),Site,n=$n, $addtags")
end

function ITensors.siteind(i::Index{Rational{Int}}; kwargs...)
  #=
    Returns the site index for a given site type
  =#
  S = parse(Rational{Int}, match(r"S=(\d+//\d+)", string(i)).captures[1])
  return siteind(SpinSiteType(S); kwargs...)
end  

function ITensors.siteind(i::Index{Rational{Int}}, n::Integer; kwargs...)
  #=
    Returns the site index for a given site type and site number
  =#
  S = parse(Rational{Int}, match(r"S=(\d+//\d+)", string(i)).captures[1])
  return siteind(SpinSiteType(S), n; kwargs...)
end 

function ITensors.siteinds(s::SpinSiteType, N::Int; kwargs...)
  #=
    Returns an array of site indices for an array of given site types
  =#
  return [siteind(s, n; kwargs...) for n in 1:N]
end

function ITensors.space(s::SpinSiteType; conserve_qns=false, conserve_sz=conserve_qns, qnname_sz="Sz")
  #=
    Returns the quantum number space for a given site
  =#
  if conserve_sz
    QNarray = Pair{QN, Int}[]
    for m in -s.S:s.S
      push!(QNarray, QN(qnname_sz, Int(2*m)) => 1)
    end
    return QNarray
  end
  return Int(2*s.S + 1)
end

function op(::OpName"Sz", s::SpinSiteType)
  #=
    Returns the Sz operator for a given site
  =#
  S = s.S
  Sz = zeros(Float64, (Int(2 * S + 1), Int(2 * S + 1)))
  for i in -S:S
    Sz[Int(i + S + 1), Int(i + S + 1)] = -i
  end
  return Sz
end

function op(::OpName"S+", s::SpinSiteType)
  #=
    Returns the S+ operator for a given site
  =#
  S = s.S
  d = Int(2 * S + 1)
  Sp = zeros(Float64, d, d)
  for i in 1:(d - 1)
      Sp[i, i + 1] = sqrt(S * (S + 1) - (-S + i - 1) * (-S + i))
  end
  return Sp
end

function op(::OpName"S-", s::SpinSiteType)
  #=
    Returns the S- operator for a given site
  =#
  return adjoint(op(OpName("S+"), s))
end

function op(::OpName"Id", s::SpinSiteType)
  #=
    Returns the identity operator for a given site
  =#
  S = s.S
  d = Int(2 * S + 1)
  return Diagonal(ones(d))
end

function ITensors.op(opname::OpName, s::SiteType)
  S_str = match(r"S=(\d+//\d+)", string(s)).captures[1]
  
  # Split the string and convert to integers, this is a workaround on daint
  num, denom = split(S_str, "//")
  numerator = parse(Int, num)
  denominator = parse(Int, denom)
  
  # Construct the Rational{Int}
  S = numerator // denominator
  
  return op(opname, SpinSiteType(S))
end

# This doesn't work on daint
# function ITensors.op(opname::OpName, s::SiteType)
#   S = parse(Rational{Int}, match(r"S=(\d+//\d+)", string(s)).captures[1])
#   return op(opname, SpinSiteType(S))
# end

function ITensors.siteinds(str::String, N::Int; kwargs...)
  if startswith(str, "S=")
    S_str = match(r"S=(\d+//\d+)", str)

    num, denom = split(S_str, "//")
    numerator = parse(Int, num)
    denominator = parse(Int, denom)

    S = numerator // denominator
    return siteinds(SpinSiteType(S), N; kwargs...)
  else
    error("Invalid siteind string: $str")
  end
end
