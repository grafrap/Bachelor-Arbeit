using ITensors

lib_dir = "."
include(lib_dir*"/Customspace.jl")

s = SpinSiteType(3//2)
sp = space(s, conserve_sz=true)

# Example call to the op method
index = Index(sp, "S=3//2,Site,n=1")
opname = OpName("Sz")
result = ITensors.op(opname, index)
println(result)