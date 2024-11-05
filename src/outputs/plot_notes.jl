using Plots

# Step 1: Read the file
file_path = "notes.txt"

if isfile(file_path)
    file_content = read(file_path, String)
    if isempty(strip(file_content))
        error("The file '$file_path' is empty.")
    end
else
    error("File '$file_path' does not exist.")
end

# Step 2: Parse the values
data = split(file_content, ";")
values = Float64[]

for s in data
    trimmed = strip(s, ['[', ']'])
    if !isempty(trimmed)
        try
            push!(values, parse(Float64, trimmed))
        catch
            println("Warning: Unable to parse '$trimmed' as Float64.")
        end
    end
end

if isempty(values)
    error("No valid Float64 data found in '$file_path'.")
end

# Step 3: Plot the values
p = plot(abs.(values), title="Values from notes.txt", xlabel="Index", ylabel="Value", legend=false, yscale=:log10)

# Step 4: Save the plot
savefig(p, "notes.png")