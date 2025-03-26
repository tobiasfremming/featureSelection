include("lookup_table.jl")
using .LookupTableModule

dataset_name = "diabetes_dataset"
dataset_extension = ".csv"
dataset_path = "datasets/$dataset_name$dataset_extension"
lut_name = "lut"
lut_extension = ".pkl"
lut_path = "../luts/$dataset_name$lut_name$lut_extension"

# Dummy fitness function
function dummy_fitness(chromosome::Vector{Bool})
    return sum(chromosome) + rand() * 0.01
end

# Create or load LUT
lut = isfile(lut_path) ? LookupTableModule.load(lut_path) : LookupTableModule.create()

# Evaluate a chromosome
chrom = [true, false, true, false, true]
fitness = LookupTableModule.get_or_evaluate!(lut, chrom, dummy_fitness)
println("Fitness: $fitness")

# Save the table
LookupTableModule.save(lut, lut_path)

# Save the table on exit
atexit(() -> LookupTableModule.save(lut, lut_path))
