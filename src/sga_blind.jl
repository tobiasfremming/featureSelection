using Random
using StatsBase
using Plots

include("lookup_table.jl")
using .LookupTableModule

using Random
using FileIO
using Dates
using FileIO
using DataFrames
using CSV

function applyMutationStandard!(population_to_mutate::Array{BitVector}, bitwiseProb::Float64)
    """
    Applies standard mutation

    Parameters:
        population_to_mutate::Array{BitVector} the population to apply mutation to
        bitwiseProb::Float64 the bitwise probability of mutation being applied
    """
    person_length = length(population_to_mutate[1])
    for person in population_to_mutate
        flipmask = rand(person_length) .< bitwiseProb
        person[flipmask] .= .!person[flipmask]
    end
end


include("GA/selection.jl")
include("forest.jl")

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

if isfile("../luts/_$(DATASET).pickle")
    TABLE::LookupTableModule.LookupTable = LookupTableModule.load("../luts/_$(DATASET).pickle")
else
    TABLE = LookupTableModule.create()
end

SAVEPATH = "out/$(Dates.format(Dates.now(), "dd.mm.HH.MM"))"
if !isdir(SAVEPATH)
    mkpath(SAVEPATH)
end

atexit(() -> LookupTableModule.save(TABLE, "../luts/_$(DATASET).pickle"))

POP_SIZE::Int = 50
NUM_PARENTS::Int = 70
NUM_ITERATIONS::Int = 500
GENE_SIZE::Int = size(DF[1], 2)
MUTATION_RATE::Float64 = 2/(GENE_SIZE + NUM_PARENTS) * 1.2
CROSSOVER_RATE::Float64 = 0.9
ELITISM_RATIO::Float64 = 0.05

# set the penalty of the zero index to be twice the maximum
ZERO_VALUE::Float64 = 1000
ZERO_KEY::String = string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))

function onePointCrossover!(population::Vector{BitVector}, prob::Float64)
    """
    Applies standard one-point crossover within contiguous pairs of children. Splits two genotypes into two partitions and matches opposite partitions for every (contiguous) pair of children.
    Operations are in-place.

    Parameters:
        population::Vector{BitVector} the vector of children to crossover. Should have an even length.
        prob::Float64 the probability of applying the crossover.
    """
    for i in 1:Int32((floor(length(population)/2)))
        parent1 = population[2*i]
        parent2 = population[2*i-1]
        if rand() < prob
            crossover_point = Int32(floor(rand()*(length(population[1]) - 1)) + 1)
            for i in 1:crossover_point
                temp = parent1[i]
                parent1[i] = parent2[i]
                parent2[i] = temp
            end
        end
    end
end

function uniformCrossover!(population::Vector{BitVector}, prob::Float64)
    """
    Applies uniform crossover within contiguous pairs of children.
    
    Parameters:
        population::Vector{BitVector} the Vector of children to crossover. Should have an even length.
        prob::Float64 the probability of applying the crossover.
    """
    for i in 1:Int32((floor(length(population)/2)))
        if rand() < prob
            parent1 = population[2*i]
            parent2 = population[2*i-1]
            for i in 1:length(population[1])
                if rand() > 0.5
                    temp = parent1[i]
                    parent1[i] = parent2[i]
                    parent2[i] = temp
                end
            end     
        end
    end
end

function initialisePopulation(nsize::Int64)::Vector{BitVector}
    """
    Creates a random and uniform population

    Parameters:
        nsize::Int64 the size of the population
    """
    population::Vector{BitVector} = Vector{BitVector}(undef, nsize)
    for i in 1:nsize
        person = bitrand(GENE_SIZE)
        population[i] = person
    end

    return population
end

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

    population::Vector = initialisePopulation(POP_SIZE)
    population_fitness::Vector{Float64} = evaluate_fitness(population)

    best_result::Union{Any, Nothing} = nothing
    best_fitness::Float64 = Inf

    for i in 1:NUM_ITERATIONS
        # parents = Selectors.rankBasedExpSelection(population, population_fitness, NUM_PARENTS, 1.0)
        parents::Vector = Selectors.fitnessProportionateSelection(population, population_fitness, NUM_PARENTS)
        children::Vector = deepcopy(parents)

        uniformCrossover!(children, CROSSOVER_RATE)
        applyMutationStandard!(children, MUTATION_RATE)

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

results = main()
df = DataFrame(
        generation = collect(1:length(results.f_mean)),
        f_mean = results.f_mean,
        f_std = results.f_std,
        f_min = results.f_min,
        f_max = results.f_max,
    )

CSV.write("$(SAVEPATH)/run.csv", df)
open("$(SAVEPATH)/best_soln.txt", "w") do f
    write(f, "$(results.best_soln)\n$(results.best_fitness)")
end
LookupTableModule.save(TABLE, "$(SAVEPATH)/_$(DATASET).pickle")

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
