# Initialize a 10x10 matrix of zeros
matrix = zeros(Int, 10, 10)

# Assign alternating values on the first off-diagonals
for i in 1:9
    value = (i % 2 == 1) ? 23 : 38
    matrix[i, i+1] = value  # Upper off-diagonal
    matrix[i+1, i] = value  # Lower off-diagonal
end

# Print the matrix
println(matrix)