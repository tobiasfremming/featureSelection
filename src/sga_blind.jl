using Random
using StatsBase
using Plots
using Random
using FileIO
using Dates
using FileIO
using DataFrames
using CSV

include("lookup_table.jl")
include("GA/selection.jl")
include("forest.jl")
include("GA/binary_mutators.jl")
include("GA/binary_crossovers.jl")
include("GA/initialisers.jl")

using .LookupTableModule
using .BinaryCrossovers
using .BinaryMutators
using .Initialisers

# parse dataset
DATASET::String = ARGS[1]
if DATASET == "cleveland"
    DF = CLEVELAND_DATASET
    FF = computeClevelandFitness
elseif DATASET == "letter"
    DF = LETTER_DATASET
    FF = computeLetterFitness
else
    DF = ZOO_DATASET
    FF = computeZooFitness
end

# load the LUT
if isfile("../luts/_$(DATASET).pickle")
    TABLE::LookupTableModule.LookupTable = LookupTableModule.load("../luts/_$(DATASET).pickle")
else
    TABLE = LookupTableModule.create()
end

# generate save folders
SAVEPATH = "out/$(Dates.format(Dates.now(), "dd.mm.HH.MM"))"
if !isdir(SAVEPATH)
    mkpath(SAVEPATH)
end

atexit(() -> LookupTableModule.save(TABLE, "../luts/_$(DATASET).pickle"))

# hyperparameters
POP_SIZE::Int = 20
NUM_PARENTS::Int = 30
NUM_ITERATIONS::Int = 500
GENE_SIZE::Int = size(DF[1], 2)
MUTATION_RATE::Float64 = 2/(GENE_SIZE + NUM_PARENTS) * 1.0
CROSSOVER_RATE::Float64 = 0.7
ELITISM_RATIO::Float64 = 0.05

# set the penalty of the zero index to be twice the maximum
ZERO_VALUE::Float64 = 1000
ZERO_KEY::String = string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))

# evaluate the fitness
function evaluate_fitness(population::Vector)::Vector{Float64}
    return map(x -> string(join(Vector{Int}(x))) == ZERO_KEY ? ZERO_VALUE : LookupTableModule.get_or_evaluate!(TABLE, x, FF), population)
end

mutable struct RunStatistics
    f_mean::Vector{Float64}
    f_std::Vector{Float64}
    f_min::Vector{Float64}
    f_max::Vector{Float64}
    best_soln::Union{Any, Nothing}
    best_fitness::Float64
end

function main()
    # statistics
    f_mean::Vector{Float64} = []
    f_std::Vector{Float64} = []
    f_min::Vector{Float64} = []
    f_max::Vector{Float64} = []

    # population structures
    population::Vector = Initialisers.initialisePopulation(POP_SIZE, GENE_SIZE)
    population_fitness::Vector{Float64} = evaluate_fitness(population)

    # storage for best results
    best_result::Union{Any, Nothing} = nothing
    best_fitness::Float64 = Inf

    for i in 1:NUM_ITERATIONS
        # selection
        parents::Vector = Selectors.fitnessProportionateSelection(population, population_fitness, NUM_PARENTS)
        children::Vector = deepcopy(parents)

        # mutation and crossover
        BinaryCrossovers.uniformCrossover!(children, CROSSOVER_RATE)
        BinaryMutators.applyMutationStandard!(children, MUTATION_RATE)

        # selection
        population = Selectors.muGammaElitistSelection(population, children, evaluate_fitness(children), population_fitness, ELITISM_RATIO)
        population_fitness = evaluate_fitness(population)

        # display some results
        print("Generation $(i) | ")
        print("Fitness $(round(mean(population_fitness), digits=3)) [$(round(minimum(population_fitness), digits=3)) - $(round(maximum(population_fitness), digits=3))] [+/- $(round(std(population_fitness), digits=3))] | ")
        print("Genotypic Diversity $(round(mean([sum(x .!= y) for x in population, y in population]), digits=3)) | ")
        print("Unique Individuals $(length(unique(population)))")
        println()

        # store results
        push!(f_mean, mean(population_fitness))
        push!(f_std, std(population_fitness))
        push!(f_min, minimum(population_fitness))
        push!(f_max, maximum(population_fitness))

        # store the best value
        if minimum(population_fitness) < best_fitness
            best_result = population[argmin(population_fitness)]
            best_fitness = minimum(population_fitness)
        end
    end

    best_bitstring::String = string(join(Vector{Int}(best_result)))
    println("Best Result: $(best_bitstring) with fitness $(best_fitness)")
    
    return RunStatistics(
        f_mean, f_std, f_min, f_max, best_result, best_fitness
    )

end

# compute results
results = main()

# log the statistics
df = DataFrame(
        generation = collect(1:length(results.f_mean)),
        f_mean = results.f_mean,
        f_std = results.f_std,
        f_min = results.f_min,
        f_max = results.f_max,
    )
CSV.write("$(SAVEPATH)/run.csv", df)

# log the best result
open("$(SAVEPATH)/best_soln.txt", "w") do f
    write(f, "$(results.best_soln)\n$(results.best_fitness)")
end

# save the lookup table
LookupTableModule.save(TABLE, "$(SAVEPATH)/_$(DATASET).pickle")

# save the run description
println("Enter a description of this run: ")
description = readline()
open("$(SAVEPATH)/index.txt", "w") do f
    write(f, "Dataset: $(DATASET)\n")
    write(f, "Description: $(description)\n")
    write(f, "Population Size: $(POP_SIZE)\n")
    write(f, "Number of Parents: $(NUM_PARENTS)\n")
    write(f, "Number of Generations: $(NUM_ITERATIONS)\n")
    write(f, "Mutation Rate: $(MUTATION_RATE)\n")
    write(f, "Crossover Rate: $(CROSSOVER_RATE)")    
end
