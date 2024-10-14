using CSV
using DataFrames
using Plots

# Function to read and process a single file
function process_file(filename)
    data = DataFrame(CSV.File(filename, delim=' ', ignorerepeated=true))
    # Extract relevant columns and calculate max time per energy step
    # Assuming columns are named appropriately after reading
    grouped_data = groupby(data, :energy)
    max_times = combine(grouped_data, :time => maximum => :max_time)
    return max_times
end

# List of filenames to process
filenames = ["job_seq_Sz0.out", "job_Sz0_pll_2.out", "job_Sz0_pll_4.out", "job_Sz0_pll_8.out", "job_Sz0_pll_16.out"]

# Process each file and store results
results = []
for filename in filenames
    push!(results, process_file(filename))
end

# Plotting the results
plot()
for (i, result) in enumerate(results)
    plot!(result.energy, result.max_time, label="File $i")
end

xlabel!("Energy Step")
ylabel!("Max Time")
title!("Max Time per Energy Step for Different Files")
display(plot)