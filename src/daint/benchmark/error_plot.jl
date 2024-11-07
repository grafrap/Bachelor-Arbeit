using DataFrames, CSV, StatsPlots, Statistics

# Read the CSV
df = CSV.read("combined_data_3000.csv", DataFrame)

# Group by script_index and N to calculate statistics over i
grouped = groupby(df, [:script_index, :N])
stats = combine(grouped,
               :Error => mean => :mean_error,
               :Error => maximum => :max_error,
               :Error => minimum => :min_error,
               :sum_χ => mean => :mean_sum,
               :sum_χ => maximum => :max_sum,
               :sum_χ => minimum => :min_sum)

# Get unique script_indices
script_indices = unique(stats.script_index)

for index in script_indices
    # Filter stats for the current script_index
    df_index = filter(row -> row.script_index == index, stats)
    
    # Plot Error statistics
    p1 = plot(df_index.N, df_index.mean_error, label="Mean Error", xlabel="N", ylabel="Error", title="Error Stats for Script $index", yscale=:log10)
    plot!(p1, df_index.N, df_index.max_error, label="Max Error")
    plot!(p1, df_index.N, df_index.min_error, label="Min Error")
    savefig(p1, "error_stats_$(index)_3000.png")
    
    # Plot sum_χ statistics
    p2 = plot(df_index.N, df_index.mean_sum, label="Mean sum_χ", xlabel="N", ylabel="sum_χ", title="sum_χ Stats for Script $index")
    plot!(p2, df_index.N, df_index.max_sum, label="Max sum_χ")
    plot!(p2, df_index.N, df_index.min_sum, label="Min sum_χ")
    hline!(p2, [2/3], label="Convergence Line")
    savefig(p2, "sum_ch_stats_$(index)_3000.png")
end