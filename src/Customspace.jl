# src/Customspace.jl
using ITensors, ITensorMPS
using LinearAlgebra

export SpinSiteType

struct SpinSiteType
  S::Rational{Int} 
end

# function tag(s::SpinSiteType) 
#   # return for 1//1 1 and for 3//2 3/2
#   double = s.S * 2
#   integ = Int(numerator(double))
#   if integ % 2 == 0
#     flo = Int(integ/2)
#     return string(flo)
#   else
#     return "$integ/2"
#   end
# end
tag(s::SpinSiteType) = s.S


function generate_spin_states(S::Rational{Int})
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
  states = generate_spin_states(S)
  state_index_map = Dict{String, Int}()
  for (i, state) in enumerate(states)
    state_index_map[state] = i
  end
  return state_index_map
end

function state_to_index(s::Index)
  index_str = string(s)
  S = match(r"S=(\d+//\d+)", index_str).captures[1]
  state_index_map = state_to_index(parse(Rational{Int}, S))
  return state_index_map
end

ITensors.state(s::SpinSiteType, n::Int) = IndexVal(n, siteind(s))
ITensors.state(s::SpinSiteType, str::String) = IndexVal(state_to_index(s.S)[str], siteind(s))
function ITensors.state(s::Index, str::String)
  state_index_map = state_to_index(s)
  arr = zeros(Int, length(state_index_map))
  arr[state_index_map[str]] = 1
  return arr
end

function ITensors.siteind(s::SpinSiteType; addtags="", kwargs...)
  sp = space(s; kwargs...)
  return Index(sp, "Site,S=$(tag(s))")
end

function ITensors.siteind(s::SpinSiteType, n::Integer; addtags="", kwargs...)
  sp = space(s; kwargs...)
  return Index(sp, "S=$(tag(s)),Site,n=$n")
end

function ITensors.siteind(i::Index{Rational{Int}}; kwargs...)
  S = parse(Rational{Int}, match(r"S=(\d+//\d+)", string(i)).captures[1])
  return siteind(SpinSiteType(S); kwargs...)
end  

function ITensors.siteind(i::Index{Rational{Int}}, n::Integer; kwargs...)
  S = parse(Rational{Int}, match(r"S=(\d+//\d+)", string(i)).captures[1])
  return siteind(SpinSiteType(S), n; kwargs...)
end 

function ITensors.siteinds(s::SpinSiteType, N::Int; kwargs...)
  return [siteind(s, n; kwargs...) for n in 1:N]
end

function ITensors.space(s::SpinSiteType; conserve_qns=false, conserve_sz=conserve_qns, qnname_sz="Sz")
  if conserve_sz
    QNarray = Pair{QN, Int}[]
    for m in -s.S:s.S
      push!(QNarray, QN(qnname_sz, Int(2*m)) => 1)
    end
    return QNarray
  end
  return 2*s.S + 1
end

function op(::OpName"Sz", s::SpinSiteType)
  S = s.S
  d = 2 * S + 1
  return Diagonal(-S:S) 
end

function op(::OpName"S+", s::SpinSiteType)
  S = s.S
  d = Int(2 * S + 1)
  Sp = zeros(d, d)
  for i in 1:(d - 1)
      Sp[i, i + 1] = sqrt(S * (S + 1) - (-S + i - 1) * (-S + i))
  end
  return Sp
end

function op(::OpName"S-", s::SpinSiteType)
  return adjoint(op(OpName("S+"), s))
end

function op(::OpName"Id", s::SpinSiteType)
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
    S = parse(Rational{Int}, split(str, "=")[2])
    return siteinds(SpinSiteType(S), N; kwargs...)
  else
    error("Invalid siteind string: $str")
  end
end
