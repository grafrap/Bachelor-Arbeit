using CSV
using DataFrames
using Plots

# Function to read and process a single file
function process_file(filename)
  # Open the file and find the line number to start reading from
  start_line = 0
  open(filename) do file
      for (i, line) in enumerate(eachline(file))
          if occursin("After sweep", line)
              start_line = i
              break
          end
      end
  end

  # Read the file starting from the identified line
  data = DataFrame(energy=Float64[], maxlinkdim=Int64[], maxerr=Float64[], time=Float64[])
  for (i, line) in enumerate(eachline(filename))
    if i < start_line
      continue
    end
    # Read the line into the dataframe
    if occursin(r"energy=\d+\.\d+", line)
      energy = parse(Float64, match(r"energy=([\d\.E\-]+)", line).captures[1])
      maxlinkdim = parse(Int64, match(r"maxlinkdim=(\d+)", line).captures[1])
      maxerr = parse(Float64, match(r"maxerr=([\d\.E\-]+)", line).captures[1])
      time = parse(Float64, match(r"time=(\d+\.\d+)", line).captures[1])
      
      # Append the extracted data to the DataFrame
      push!(data, (energy, maxlinkdim, maxerr, time))
    end
  end
 
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


min_energy = minimum([minimum(result[!, :energy]) for result in results])

global processes = 1
# Plotting the results
plot(size=(800, 600))
for result in results
  global processes

  cumulative_time = 0
  cumulative_times = []
  energies = []
  for row in eachrow(result)
    cumulative_time += row[:max_time]
    push!(cumulative_times, cumulative_time)
    push!(energies, row[:energy] - min_energy)
  end
  plot!(cumulative_times, energies, yscale=:log10, xscale=:log2, label="$processes Processes")
  processes *= 2
  println(cumulative_times[end])
end
#change layout of plot


yticks!(10.0 .^(-4:1:4), ["1e-4","1e-3", "1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3", "1e4"])
ylims!(1e-4, 1e4)
xticks!(10.0 .^(0:1:4), ["1", "10", "100", "1000", "10000"])
# plot(cumulative_times, energies, label="Strong Scaling", yscale=:log10)
xlabel!("Time [s]")
ylabel!("Energy (difference to minimum) [kJ or eV?]")
title!("Max Time per Energy Step for Different number of Processes")
savefig("strong_scaling.pdf")