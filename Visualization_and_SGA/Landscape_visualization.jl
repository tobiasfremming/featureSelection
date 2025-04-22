# using TSne, Statistics, MLDatasets

# rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())

# alldata, allabels = MNIST.traindata(Float64);
# data = reshape(permutedims(alldata[:, :, 1:2500], (3, 1, 2)),
#                2500, size(alldata, 1)*size(alldata, 2));
# # Normalize the data, this should be done if there are large scale differences in the dataset
# X = rescale(data, dims=1);

# Y = tsne(X, 2, 50, 1000, 20.0);

# using Plots
# theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(allabels[1:size(Y,1)]))
# Plots.pdf(theplot, "myplot.pdf")


using TSne
using Plots
using CSV
using DataFrames

# Load CSV file (no headers)
df = CSV.read("Lookup_tables/diabetes_err.csv", DataFrame, header=false)

# Extract feature bitstrings (first column) and error values (second column)
feature_bitstrings = df[:, 1]  # First column (binary feature selection stored as Int)
errors = df[:, 2]  # Second column (corresponding model errors)

# Generate integers from 0 to 255 (all possible combinations for 8 bits)
n_combinations = 256  # From 0 to 255
bit_length = 8  # Since you're working with 8-bit binary vectors

# Convert each integer to a binary vector of length 8
function int_to_binary_vector(n, bit_length)
    bin_str = lpad(string(n, base=2), bit_length, '0')  # Convert integer to binary string, padding to 8 bits
    return parse.(Int, collect(bin_str))  # Convert each binary digit (character) to an integer
end

# Generate the matrix X by applying int_to_binary_vector to all integers from 0 to 255
X = hcat([int_to_binary_vector(i, bit_length) for i in 0:(n_combinations - 1)]...)'
X = Matrix{Number}(X)

# Apply t-SNE to reduce dimensionality to 2D
tsne_result = tsne(X, 2, 1000, 30)  # (data, target_dims, iterations, perplexity)

# Extract 2D coordinates
x_tsne = tsne_result[:, 1]
y_tsne = tsne_result[:, 2]
z_errors = errors

plotly()
# Create 3D scatter plot
# Create a basic 3D scatter plot without coloring
# scatter3d(x_tsne, y_tsne, errors, 
#     xlabel="t-SNE Dim 1", ylabel="t-SNE Dim 2", zlabel="Error Value", 
#     title="t-SNE Feature Landscape with Errors")

# Sort the coordinates to better form a grid
sorted_indices = sortperm(x_tsne)
x_sorted = x_tsne[sorted_indices]
y_sorted = y_tsne[sorted_indices]
z_sorted = z_errors[sorted_indices]

# Create a 2D grid (meshgrid) for surface plotting
using LinearAlgebra
x_range = LinRange(minimum(x_sorted), maximum(x_sorted), length(x_sorted))
y_range = LinRange(minimum(y_sorted), maximum(y_sorted), length(y_sorted))

# Create meshgrid for x and y
X_mesh, Y_mesh = meshgrid(x_range, y_range)

# Interpolate over the grid
# We will use `interpolate` function from Interpolations.jl
interp = interpolate((x_sorted, y_sorted), z_sorted, Gridded(Linear()))

# Interpolate the z values on the meshgrid
Z_mesh = [interp(x, y) for (x, y) in zip(vec(X_mesh), vec(Y_mesh))]

# Plot the surface
surface(X_mesh, Y_mesh, Z_mesh, 
    xlabel="t-SNE Dim 1", ylabel="t-SNE Dim 2", zlabel="Error Value", 
    title="t-SNE Feature Landscape as Surface")

function meshgrid(x, y)
    X = repeat(x, length(y), 1)
    Y = repeat(y, length(x), 1)'
    return X, Y
end