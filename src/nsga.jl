using Random
using StatsBase
using Plots

include("lookup_table.jl")
using .LookupTableModule

DATASET = "heart"
ERROR_TABLE = LookupTableModule.load("../luts/$(DATASET)_err.pickle")
PEN_TABLE = LookupTableModule.load("../luts/$(DATASET)_pen.pickle")

POP_SIZE = 50
NUM_PARENTS = 50
NUM_ITERATIONS = 100
GENE_SIZE = length(collect(keys(ERROR_TABLE.table))[1])

ERROR_TABLE.table[string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))] = 100
PEN_TABLE.table[string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))] = 100

function onePointCrossover!(population::Array{BitVector}, prob::Float64)
    """
    Applies standard one-point crossover within contiguous pairs of children. Splits two genotypes into two partitions and matches opposite partitions for every (contiguous) pair of children.
    Operations are in-place.

    Parameters:
        population::Array{BitVector} the array of children to crossover. Should have an even length.
        prob::Float32 the probability of applying the crossover.
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
        prob::Float32 the probability of applying the crossover.
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

function applyMutationStandard!(population_to_mutate::Array{BitVector}, bitwiseProb::Float64)
    """
    Applies standard mutation

    Parameters:
        population_to_mutate::Array{BitVector} the population to apply mutation to
        bitwiseProb::Float32 the bitwise probability of mutation being applied
    """
    for person in population_to_mutate
        for i in 1:length(population_to_mutate[1])
            if rand() < bitwiseProb
                person[i] = 1 - person[i]
            end
        end
    end
end

function get_population_ranks(fitness_one::Array{Float64}, fitness_two::Array{Float64})
    N = length(fitness_one)
    population_ranks = fill(1, N)

    for i in 1:N
        for j in 1:N
            if i != j
                dominates_i = (fitness_one[i] <= fitness_one[j] && fitness_two[i] <= fitness_two[j]) && 
                                (fitness_one[i] < fitness_one[j] || fitness_two[i] < fitness_two[j])

                dominates_j = (fitness_one[j] <= fitness_one[i] && fitness_two[j] <= fitness_two[i]) && 
                                (fitness_one[j] < fitness_one[i] || fitness_two[j] < fitness_two[i])

                if dominates_i
                    population_ranks[j] = max(population_ranks[j], population_ranks[i] + 1)
                elseif dominates_j
                    population_ranks[i] = max(population_ranks[i], population_ranks[j] + 1)
                end
            end
        end
    end

    return population_ranks
end

function get_population_crowdings(fitness_one::Array{Float64}, fitness_two::Array{Float64}, ranks)
    crowdings = fill(0.0, length(ranks))

    # do crowding rank-wise
    for rank in 1:maximum(ranks)
        rank_set = findall(x -> x == rank, ranks)
        first_fitness = fitness_one[rank_set]
        second_fitness = fitness_two[rank_set]

        sorted_by_first = sortperm(first_fitness, rev=true)
        sorted_by_second = sortperm(second_fitness, rev=true)

        # set all extrema to be infinite
        crowdings[rank_set[sorted_by_first[1]]] = Inf
        crowdings[rank_set[sorted_by_second[1]]] = Inf
        crowdings[rank_set[sorted_by_first[end]]] = Inf
        crowdings[rank_set[sorted_by_second[end]]] = Inf

        # calculate bounds
        min_first = first_fitness[sorted_by_first[1]]
        max_first = first_fitness[sorted_by_first[end]]
        min_second = second_fitness[sorted_by_second[1]]
        max_second = second_fitness[sorted_by_second[end]]

        # for all others, do this:
        for (i, ele) in enumerate(sorted_by_first)
            if i == 1 || i == length(sorted_by_first) 
                crowdings[rank_set[ele]] = Inf
            else
                crowdings[rank_set[ele]] += (
                    (first_fitness[sorted_by_first[i + 1]] - first_fitness[sorted_by_first[i - 1]]) / 
                    (max_first - min_first)
                )
            end
        end
        for (i, ele) in enumerate(sorted_by_second)
            if i == 1 || i == length(sorted_by_second) 
                crowdings[rank_set[ele]] = Inf
            else
                crowdings[rank_set[ele]] += (
                    (second_fitness[sorted_by_second[i + 1]] - second_fitness[sorted_by_second[i - 1]]) / 
                    (max_second - min_second)
                )
            end
        end
    end

    return crowdings
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

function evaluate_fitness_one(population)::Array{Float64}
    return map(x -> ERROR_TABLE[string(join(Vector{Int}(x)))], population)
end

function evaluate_fitness_two(population)::Array{Float64}
    return map(x -> PEN_TABLE[string(join(Vector{Int}(x)))], population)
end

function parentSelectionNSGA(population::Vector, nparents::Int)::Array
    """
    Uses a binary tournament
    """
    new_population::Array = Vector(undef, nparents)

    fitness_one = evaluate_fitness_one(population)
    fitness_two = evaluate_fitness_two(population)
    old_pareto_ranks::Array{Int} = get_population_ranks(fitness_one, fitness_two)
    old_crowding_distances::Array{Float64} = get_population_crowdings(fitness_one, fitness_two, old_pareto_ranks)
    
    for i in 1:nparents
        # choose two random members
        option_one, option_two = sample(1:length(population), 2, replace=false)

        if old_pareto_ranks[option_one] < old_pareto_ranks[option_two]
            new_population[i] = deepcopy(population[option_one])
        elseif old_pareto_ranks[option_one] > old_pareto_ranks[option_two]
            new_population[i] = deepcopy(population[option_two])
        else
            new_population[i] = old_crowding_distances[option_one] > old_crowding_distances[option_two] ? deepcopy(population[option_one]) : deepcopy(population[option_two])
        end
    end

    return new_population
end

function survivorSelectionNSGA(parents::Array, children::Array)::Array
    N = length(parents)
    total_population = vcat(parents, children)
    fitness_one = evaluate_fitness_one(total_population)
    fitness_two = evaluate_fitness_two(total_population)

    total_ranks = get_population_ranks(fitness_one, fitness_two)
    total_crowdings = get_population_crowdings(fitness_one, fitness_two, total_ranks)

    output_population = []

    rank = 1
    while rank <= maximum(total_ranks)
        next_front = findall(x -> x == rank, total_ranks)
        if length(output_population) + length(next_front) <= N
            append!(output_population, total_population[next_front])
        else
            next_crowdings = total_crowdings[next_front]
            crowding_indices = sortperm(next_crowdings, rev=true)[1:N - length(output_population)]
            append!(output_population, total_population[next_front[crowding_indices]])
        end
        rank += 1
    end

    return deepcopy(output_population)
end


function main()

    population::Array{BitVector} = initialisePopulation(POP_SIZE)

    for i in 1:NUM_ITERATIONS
        new_fitness_one = evaluate_fitness_one(population)
        new_fitness_two = evaluate_fitness_two(population)

        print("Generation $(i) | ")
        print("FitnessOne $(round(mean(new_fitness_one), digits=3)) [$(round(minimum(new_fitness_one), digits=3)) - $(round(maximum(new_fitness_one), digits=3))] [+/- $(round(std(new_fitness_one), digits=3))] | ")
        print("FitnessTwo $(round(mean(new_fitness_two), digits=3)) [$(round(minimum(new_fitness_two), digits=3)) - $(round(maximum(new_fitness_two), digits=3))] [+/- $(round(std(new_fitness_two), digits=3))] | ")
        print("Genotypic Diversity $(round(mean([sum(x .!= y) for x in population, y in population]), digits=3)) | ")
        println()

        plot(
            new_fitness_one, new_fitness_two, 
            seriestype=:scatter,
            color=:viridis,
            marker_z=get_population_ranks(new_fitness_one, new_fitness_two)
        )
        savefig("out/landscape_$(i).png")

        parents = parentSelectionNSGA(population, NUM_PARENTS)
        children::Array{BitVector} = deepcopy(parents)
        uniformCrossover!(children, 0.6)
        applyMutationStandard!(children, 1/30)

        population = survivorSelectionNSGA(parents, children)
    end


end

main()