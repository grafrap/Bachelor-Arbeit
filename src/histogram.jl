using DelimitedFiles
using Plots
using Printf
# Julia code to read chi.txt as a matrix of floats

function read_file_as_matrix(file_path)
  matrix = []
  open(file_path, "r") do file
      # Read only the first line
      first_line = readline(file)
      content = strip(first_line, ['[', ']'])
      content = replace(content, ']' => "")
      rows = split(content, ';')
      for row in rows
          # Remove any trailing semicolon and whitespace, then split by spaces
          float_values = split(strip(row))
          # Convert each value to Float64 and append to the matrix
          push!(matrix, parse.(Float64, float_values))
      end
  end
  return hcat(matrix...)
end

# Read in the files
file_path = "outputs/chi_values.txt"
matrix = read_file_as_matrix(file_path)
# println(matrix)


# for col in 1:size(matrix, 2)
#   matrix[:, col] .-= minimum(matrix[:, col])
#   matrix[:, col] ./= maximum(matrix[:, col])
# end
# Normalize the matrix
min = minimum(matrix)
matrix = matrix .- min
max = maximum(matrix)
matrix = matrix ./ max

# for row in eachrow(matrix)
#   for val in row
#     @printf("%.3e ", val)
#   end
#   println()
# end

# Use the heatmap function to create the 2D histogram
heatmap(matrix, xlabel="Sites", ylabel="frequencies", title="2D Histogram of Matrix for N = 1000", size=(800, 600), margin=10Plots.mm, color=:viridis)

# Save the plot to a file
savefig("histograms/2d_histogram_8_false_deltatest_1000.png")