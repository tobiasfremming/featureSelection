using Random
using StatsBase
using Plots

include("lookup_table.jl")
using .LookupTableModule

include("GA/mutate.jl")

DATASET::String = ARGS[1]
ERROR_TABLE::LookupTableModule.LookupTable = LookupTableModule.load("../luts/$(DATASET)_err.pickle")
PEN_TABLE::LookupTableModule.LookupTable = LookupTableModule.load("../luts/$(DATASET)_pen.pickle")

POP_SIZE = 200
NUM_PARENTS = 200
NUM_ITERATIONS = 100
GENE_SIZE = length(collect(keys(ERROR_TABLE))[1])
MUTATION_RATE::Float64 = 2/(GENE_SIZE + NUM_PARENTS) * 1.0
CROSSOVER_RATE::Float64 = 0.7

# remove zero index of the table
ZERO_KEY::String = string(join(Vector{Int}([0 for _ in 1:GENE_SIZE])))
if haskey(ERROR_TABLE, ZERO_KEY)
    delete!(ERROR_TABLE, ZERO_KEY)
end
if haskey(PEN_TABLE, ZERO_KEY)
    delete!(PEN_TABLE, ZERO_KEY)
end

# set the penalty of the zero index to be the maximum
ZERO_VALUE_F1::Float64 = maximum(map(x -> ERROR_TABLE[x], collect(keys(ERROR_TABLE)))) + 1
ZERO_VALUE_F2::Float64 = maximum(map(x -> PEN_TABLE[x], collect(keys(PEN_TABLE)))) + 1

function hypervolume(pareto_front, reference_point)
    # Sort points by the first objective (minimization assumed)
    sorted_front = sort(pareto_front, by=x -> x[1], rev=true)

    hv = 0.0
    prev_x = reference_point[1]

    for (x, y) in sorted_front
        hv += (reference_point[2] - y) * (prev_x - x)
        prev_x = x
    end

    return hv
end

function onePointCrossover!(population::Vector{BitVector}, prob::Float64)
    """
    Applies standard one-point crossover within contiguous pairs of children. Splits two genotypes into two partitions and matches opposite partitions for every (contiguous) pair of children.
    Operations are in-place.

    Parameters:
        population::Vector{BitVector} the Vector of children to crossover. Should have an even length.
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

function get_population_ranks(fitness_one::Vector{Float64}, fitness_two::Vector{Float64})
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

function get_population_crowdings(fitness_one::Vector{Float64}, fitness_two::Vector{Float64}, ranks)
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

function evaluate_fitness_one(population)::Vector{Float64}
    return map(x -> string(join(Vector{Int}(x))) == ZERO_KEY ? ZERO_VALUE_F1 : ERROR_TABLE[string(join(Vector{Int}(x)))], population)
end

function evaluate_fitness_two(population)::Vector{Float64}
    return map(x -> string(join(Vector{Int}(x))) == ZERO_KEY ? ZERO_VALUE_F2 : PEN_TABLE[string(join(Vector{Int}(x)))], population)
end

function parentSelectionNSGA(population::Vector, nparents::Int)::Vector
    """
    Uses a binary tournament
    """
    new_population::Vector = Vector(undef, nparents)

    fitness_one::Vector{Float64} = evaluate_fitness_one(population)
    fitness_two::Vector{Float64} = evaluate_fitness_two(population)
    old_pareto_ranks::Vector{Int} = get_population_ranks(fitness_one, fitness_two)
    old_crowding_distances::Vector{Float64} = get_population_crowdings(fitness_one, fitness_two, old_pareto_ranks)
    
    for i in 1:nparents
        # choose two random members
        option_one::Int, option_two::Int = sample(1:length(population), 2, replace=false)

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

function survivorSelectionNSGA(parents::Vector, children::Vector)::Vector
    N::Int = length(parents)
    total_population::Vector = vcat(parents, children)
    fitness_one::Vector{Float64} = evaluate_fitness_one(total_population)
    fitness_two::Vector{Float64} = evaluate_fitness_two(total_population)

    total_ranks::Vector{Int} = get_population_ranks(fitness_one, fitness_two)
    total_crowdings::Vector{Float64} = get_population_crowdings(fitness_one, fitness_two, total_ranks)

    output_population::Vector = []

    rank::Int = 1
    while rank <= maximum(total_ranks)
        next_front::Vector{Int} = findall(x -> x == rank, total_ranks)
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

mutable struct Statistics
    f2_mean::Vector{Float64}
    f2_std::Vector{Float64}
    f2_min::Vector{Float64}
    f2_max::Vector{Float64}
    f1_mean::Vector{Float64} 
    f1_std::Vector{Float64}
    f1_min::Vector{Float64}
    f1_max::Vector{Float64}
    percent_front::Vector{Float64}
    percent_dom::Vector{Float64}
    percent_hyperv::Vector{Float64}
    best_front_percent::Float64
    best_hyperv_percent::Float64
end


function main(suppress_plots=false)

    population::Vector{BitVector} = initialisePopulation(POP_SIZE)

    all_points_err::Vector{Float64} = collect(values(ERROR_TABLE))
    all_points_pen::Vector{Float64} = collect(values(PEN_TABLE))
    total_population = [digits(i, base=2, pad=GENE_SIZE) for i in 1:2^(GENE_SIZE)]
    total_pareto_indices = findall(x -> x == 1, get_population_ranks(all_points_err, all_points_pen))
    total_pareto_front::Vector = total_population[total_pareto_indices]
    total_pareto_front_values = collect(zip(all_points_err[total_pareto_indices], all_points_pen[total_pareto_indices]))
    total_pareto_front_values = unique(total_pareto_front_values)
    hypervolume_ref_point = [maximum(all_points_err), maximum(all_points_pen)]
    reference_hypervolume = round(hypervolume(total_pareto_front_values, hypervolume_ref_point), digits=3)

    for i in 1:NUM_ITERATIONS
        parents::Vector = parentSelectionNSGA(population, NUM_PARENTS)
        children::Vector{BitVector} = deepcopy(parents)
        uniformCrossover!(children, CROSSOVER_RATE)
        Mutations.applyMutationStandard!(children, MUTATION_RATE)

        population = survivorSelectionNSGA(parents, children)

        new_fitness_one::Vector{Float64} = evaluate_fitness_one(population)
        new_fitness_two::Vector{Float64} = evaluate_fitness_two(population)

        pareto_front_indices = findall(x -> x == 1, get_population_ranks(new_fitness_one, new_fitness_two))
        pareto_front_values = collect(zip(new_fitness_one[pareto_front_indices], new_fitness_two[pareto_front_indices]))
        pareto_front_values = unique(pareto_front_values)

        print("Generation $(i) | ")
        print("F1 $(round(mean(new_fitness_one), digits=3)) [$(round(minimum(new_fitness_one), digits=3)) - $(round(maximum(new_fitness_one), digits=3))] [+/- $(round(std(new_fitness_one), digits=3))] | ")
        print("F2 $(round(mean(new_fitness_two), digits=3)) [$(round(minimum(new_fitness_two), digits=3)) - $(round(maximum(new_fitness_two), digits=3))] [+/- $(round(std(new_fitness_two), digits=3))] | ")
        print("G Div $(round(mean([sum(x .!= y) for x in population, y in population]), digits=3)) | ")
        print("% of Front $(length(setdiff(total_pareto_front, population))/length(total_pareto_front)*100) | ")
        print("% of Dominated Solns $(length(setdiff(population, total_pareto_front))/length(population)*100) | ")
        print("Hypervolume $(round(hypervolume(pareto_front_values, hypervolume_ref_point), digits=3))/$(reference_hypervolume)")
        println()
        
        if !suppress_plots
        plot(
            all_points_err, all_points_pen,
            seriestype=:scatter,
            color=:gray,
            alpha=0.1,
            label="Entire Table",
            legend=:topright
        )
        plot!(
            new_fitness_one, new_fitness_two, 
            seriestype=:scatter,
            color=:viridis,
            marker_z=get_population_ranks(new_fitness_one, new_fitness_two),
            label="Population"
        )
        savefig("out/landscape_$(i).png")

        plot(
            new_fitness_one, new_fitness_two, 
            seriestype=:scatter,
            color=:viridis,
            marker_z=get_population_ranks(new_fitness_one, new_fitness_two),
            label="Population"
        )
        savefig("out/landscapeonly_$(i).png")
        end
    end
end

main(true)