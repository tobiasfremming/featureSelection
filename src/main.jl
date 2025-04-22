include("lookup_table.jl")
include("GA/agent.jl")
include("GA/crossover.jl")
include("GA/mutate.jl")
include("GA/beat_selection.jl")

using .LookupTableModule
using .Agent
using .Mutations

dataset_name = "diabetes_dataset"
dataset_extension = ".csv"
dataset_path = "datasets/$dataset_name$dataset_extension"
lut_name = "lut"
lut_extension = ".pkl"
lut_path = "../luts/$dataset_name$lut_name$lut_extension"

# Dummy fitness function
# function dummy_fitness(chromosome::Agent.Chromosome)
#     # This is a placeholder. Replace with actual fitness evaluation logic.
#     # Example: sum of gene weights divided by number of genes
#     gene_sum = 0.0
#     for gene in chromosome.genes
#         gene_sum += gene.weight
#     end
#     return sum(gene_sum/ length(chromosome.genes))
# end

function dummy_fitness(bits::Vector{Char})
    # return average number of 1s
    return sum(bit == '1' ? 1 : 0 for bit in bits) / length(bits)
    
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
    for chromosome::Agent.Chromosome in population
        # Evaluate fitness
        bit_vector = Agent.forward(chromosome)
        chromosome.fitness = dummy_fitness(bit_vector)
    end
    species = Agent.speciate_and_fitness_sharing!(population, c1, c2, c3, delta_t)
    
    # Sort by fitness
    sort!(population, by = x -> -x.adjusted_fitness)

    # Elitism
    population_size = length(population)
    elite_count = Int(round(population_size * elite_fraction))
    new_population = deepcopy(population[1:elite_count])

    # Fill the rest
    #while length(new_population) < length(population)
        #parent1 = BeatSelection.select_parents_nsga(population)
        #parent2 = BeatSelection.select_parents_nsga(population)
    parents = select_parents_nsga(population, population_size)

    for i in 1:2:length(parents)-1
        parent1 = parents[i]
        parent2 = parents[i+1]
        child = crossover(parent1, parent2)
        Mutations.mutate!(child, 
            prob_weight_perturb=0.8,
            prob_weight_reset=0.3,
            prob_toggle=0.1,
            prob_add_gene=0.05,
            prob_add_node=0.03,
            max_node_id=100
        )
        push!(new_population, child)
    end

  

    # Replace old population
    empty!(population)
    append!(population, new_population)
end

function main()
    # Load dataset

    # Initialize population
    population_size = 100
    population = [Agent.create_random_chromosome(10, 0, 1) for _ in 1:population_size]

    # Parameters
    c1 = 1.0
    c2 = 1.0
    c3 = 1.0
    delta_t = 0.5
    elite_fraction = 0.5

    # Run generations
    for generation in 1:100
        run_generation!(population, c1, c2, c3, delta_t, elite_fraction)
        println("Generation $generation: Best fitness: $(population[1].fitness)")
    end

end

main()