using DelimitedFiles
using Plots
using Printf
# Julia code to read chi.txt as a matrix of floats

function read_file_as_matrix(file_path)
    matrix = []
    open(file_path, "r") do file
      content = strip(read(file, String), ['[', ']'])
      content = replace(content, ']' => "")
      rows = split(content, ';')
        for row in rows
            # Remove any trailing semicolon and whitespace, then split by spaces
            float_values = split(strip(row))
            # Convert each value to float and append to the matrix
            push!(matrix, parse.(Float64, float_values))
        end
    end
    return hcat(matrix...)
end

# Example usage
file_path = "outputs/chi_values.txt"
matrix = read_file_as_matrix(file_path)
println(matrix)


# Normalize each column by subtracting the min and divide by the max
for col in 1:size(matrix, 2)
  min = minimum(matrix[:, col])
  matrix[:, col] .= matrix[:, col] .- min
  max = maximum(matrix[:, col])
  matrix[:, col] .= matrix[:, col] ./ max
end

# for row in 1:size(matrix, 1)
#   min = minimum(matrix[row, :])
#   matrix[row, :] .= matrix[row, :] .- min
#   max = maximum(matrix[row, :])
#   matrix[row, :] .= matrix[row, :] ./ max
# end



for row in eachrow(matrix)
  for val in row
    @printf("%.3e ", val)
  end
  println()
end

# Step 4: Use the heatmap function to create the 2D histogram
heatmap(matrix, xlabel="Sites", ylabel="frequencies", title="2D Histogram of Matrix", size=(800, 600), margin=10Plots.mm, color=:viridis)

# Save the plot to a file
savefig("2d_histogram.png")