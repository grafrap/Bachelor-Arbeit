using DelimitedFiles
using Plots
using Printf
# Julia code to read chi.txt as a matrix of floats

function read_file_as_matrix(file_path)
    matrix = []
    open(file_path, "r") do file
        for line in eachline(file)
            # Remove any trailing semicolon and whitespace, then split by spaces
            float_values = split(strip(replace(line, ";" => "")))
            # Convert each value to float and append to the matrix
            push!(matrix, parse.(Float64, float_values))
        end
    end
    return hcat(matrix...)
end

# Example usage
file_path = "chi.txt"
matrix = read_file_as_matrix(file_path)
println(matrix)


# Normalize each column by subtracting the min and divide by the max
for row in 1:size(matrix, 1)
  min = minimum(matrix[row, :])
  matrix[row, :] .= matrix[row, :] .- min
  max = maximum(matrix[row, :])
  matrix[row, :] .= matrix[row, :] ./ max
end

for row in eachrow(matrix)
  for val in row
    @printf("%.3e ", val)
  end
  println()
end

# Step 4: Use the heatmap function to create the 2D histogram
heatmap(matrix, xlabel="Sites", ylabel="frequencies", title="2D Histogram of Matrix")

# Save the plot to a file
savefig("2d_histogram.png")