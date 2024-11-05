using CSV
using DataFrames
using Plots
using Glob
using FilePaths

# Read the file
file_path = "daint_conv_6000_om100.txt"
data = DataFrame(N=Int[], i=Int[], Error=Float64[])

open(file_path, "r") do file
    for line in eachline(file)
        if startswith(line, "N =")
            parts = split(line, r"[,\t= ]+")
            push!(data, (
                N = parse(Int, parts[2]),
                i = parse(Int, parts[4]),
                Error = parse(Float64, parts[6])
            ))
        end
    end
end

# Create convergence directory
isdir("convergence") || mkdir("convergence")

# Plot for each i
unique_i = sort(unique(data.i))
mygif = @animate for current_i in unique_i
    subset_data = filter(row -> row.i == current_i, data)
    plot(subset_data.N, subset_data.Error, xlabel="N", ylabel="Error",
         title="N vs Error for i = $current_i", legend=false, yscale=:log10,
         yticks=([0.001, 0.002, 0.003, 0.004, 0.005, 
         0.01, 0.02, 0.05, 0.1, 0.2, 
         0.5, 1.0], string.([0.001, 0.002, 0.003, 0.004, 0.005, 
                                            0.01, 0.02, 0.05, 0.1, 0.2, 
                                            0.5, 1.0])))
    savefig("convergence/convergence_$(current_i)_N=6000_100.pdf")
end

gif(mygif, "convergence/convergence_6000_100.gif", fps=0.5)