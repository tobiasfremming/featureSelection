using Random
using StatsBase
using Plots

include("lookup_table.jl")
using .LookupTableModule

include("GA/mutate.jl")
include("GA/selection.jl")

DATASET = "cancer"
ERROR_TABLE = LookupTableModule.load("../luts/$(DATASET)_err.pickle")
PEN_TABLE = LookupTableModule.load("../luts/$(DATASET)_pen.pickle")

LAMBDA = 0.0 # regularisation constant (penalty factor)
POP_SIZE = 500
NUM_PARENTS = 700
NUM_ITERATIONS = 1000
GENE_SIZE = length(collect(keys(ERROR_TABLE))[1])
MUTATION_RATE = 2/(GENE_SIZE + NUM_PARENTS) * 1.0
CROSSOVER_RATE = 0.7
ELITISM_RATIO = 0.05

# remove zero index
ZERO_KEY = string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))
if haskey(ERROR_TABLE, ZERO_KEY)
    delete!(ERROR_TABLE, ZERO_KEY)
end
if haskey(PEN_TABLE, ZERO_KEY)
    delete!(PEN_TABLE, ZERO_KEY)
end

# set zero penalty
ZERO_VALUE = maximum(2*map(x -> ERROR_TABLE[x] + LAMBDA*PEN_TABLE[x], collect(keys(ERROR_TABLE))))

function onePointCrossover!(population::Array{BitVector}, prob::Float64)
    """
    Applies standard one-point crossover within contiguous pairs of children. Splits two genotypes into two partitions and matches opposite partitions for every (contiguous) pair of children.
    Operations are in-place.

    Parameters:
        population::Array{BitVector} the array of children to crossover. Should have an even length.
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

function uniformCrossover!(population::Array{BitVector}, prob::Float64)
    """
    Applies uniform crossover within contiguous pairs of children.
    
    Parameters:
        population::Array{BitVector} the array of children to crossover. Should have an even length.
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

function initialisePopulation(nsize::Int64)::Array{BitVector}
    """
    Creates a random and uniform population

    Parameters:
        nsize::Int64 the size of the population
    """
    population::Array{BitVector} = Array{BitVector}(undef, nsize)
    for i in 1:nsize
        person = bitrand(GENE_SIZE)
        population[i] = person
    end

    return population
end

function evaluate_fitness(population)::Array{Float64}
    return map(x -> string(join(Vector{Int}(x))) == ZERO_KEY ? ZERO_VALUE : ERROR_TABLE[string(join(Vector{Int}(x)))] + LAMBDA*PEN_TABLE[string(join(Vector{Int}(x)))], population)
end

function main()

    population::Array = initialisePopulation(POP_SIZE)
    population_fitness::Array{Float64} = evaluate_fitness(population)

    best_result::Union{Any, Nothing} = nothing
    best_fitness::Float64 = Inf

    for i in 1:NUM_ITERATIONS
        # parents = Selectors.rankBasedExpSelection(population, population_fitness, NUM_PARENTS, 1.0)
        parents = Selectors.fitnessProportionateSelection(population, population_fitness, NUM_PARENTS)
        children::Array{BitVector} = deepcopy(parents)
        uniformCrossover!(children, CROSSOVER_RATE)
        Mutations.applyMutationStandard!(children, MUTATION_RATE)

        population = Selectors.muGammaElitistSelection(population, children, evaluate_fitness(children), population_fitness, ELITISM_RATIO)

        population_fitness = evaluate_fitness(population)

        print("Generation $(i) | ")
        print("Fitness $(round(mean(population_fitness), digits=3)) [$(round(minimum(population_fitness), digits=3)) - $(round(maximum(population_fitness), digits=3))] [+/- $(round(std(population_fitness), digits=3))] | ")
        print("Genotypic Diversity $(round(mean([sum(x .!= y) for x in population, y in population]), digits=3)) | ")
        print("Unique Individuals $(length(unique(population)))")
        println()

        # store the best value
        if minimum(population_fitness) < best_fitness
            best_result = population[argmin(population_fitness)]
            best_fitness = minimum(population_fitness)
        end
    end

    best_bitstring = string(join(Vector{Int}(best_result)))
    println("Best Result: $(best_bitstring) with fitness $(best_fitness), penalty $(PEN_TABLE[best_bitstring]), error $(ERROR_TABLE[best_bitstring])")


end

main()