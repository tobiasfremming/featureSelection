include("lookup_table.jl")
include("GA/agent.jl")
include("GA/crossover.jl")
include("GA/mutate.jl")
include("GA/beat_selection.jl")

using .LookupTableModule, .Agent, .Crossover, .Mutations, .BeatSelection

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
# lut = isfile(lut_path) ? LookupTableModule.load(lut_path) : LookupTableModule.create()

# Evaluate a chromosome
# chrom = [true, false, true, false, true]
# fitness = LookupTableModule.get_or_evaluate!(lut, chrom, dummy_fitness)
# println("Fitness: $fitness")

# # Save the table
# LookupTableModule.save(lut, lut_path)

# Save the table on exit
# atexit(() -> LookupTableModule.save(lut, lut_path))

function run_generation!(population::Vector{Agent.Chromosome}, c1, c2, c3, delta_t, elite_fraction)
    species = Agent.speciate_and_fitness_sharing!(population, c1, c2, c3, delta_t)
    
    # Sort by fitness
    sort!(population, by = x -> -x.adjusted_fitness)

    # Elitism
    elite_count = Int(round(length(population) * elite_fraction))
    new_population = deepcopy(population[1:elite_count])

    # Fill the rest
    while length(new_population) < length(population)
        parent1 = select_parent(population)
        parent2 = select_parent(population)
        child = Crossover.crossover(parent1, parent2)
        mutate!(child)
        push!(new_population, child)
    end

    # Replace old population
    empty!(population)
    append!(population, new_population)
end
