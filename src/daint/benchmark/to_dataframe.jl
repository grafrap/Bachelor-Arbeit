using DataFrames, CSV

df = DataFrame(script_index=Int[], N=Int[], i=Int[], Error=Float64[], sum_χ=Float64[], Δsum=Float64[], Δt=Int[])

for index in 50:50:1000
  filename = "bench_$(index)_N=3000.err"
    open(filename, "r") do file
    for line in eachline(file)
      if startswith(line, "N =")
        parts = split(line, r"\s+")
        N = parse(Int, parts[3])
        i = parse(Int, parts[6])
        Error = parse(Float64, parts[9])
        sum_χ = parse(Float64, parts[12])
        Δsum = parse(Float64, parts[15])
        Δt = parse(Int, replace(parts[18], r"[^\d]" => ""))
        push!(df, (script_index=index, N=N, i=i, Error=Error, sum_χ=sum_χ, Δsum=Δsum, Δt=Δt))
      end
    end
  end
end

CSV.write("combined_data_3000.csv", df)