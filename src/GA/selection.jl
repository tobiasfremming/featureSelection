module Selectors

using StatsBase

function fitnessProportionateSelection(population::Array, population_fitness::Array{Float64}, num_parents::Int64)::Array
    """
    Selects the parents based on a weighted lottery
    """
    population_fitness_corrected::Array{Float64} = 1 .- population_fitness./sum(filter(x->!isinf(x), population_fitness))
    population_fitness_corrected = map(x -> isinf(x) ? 0 : x, population_fitness_corrected)
    parents::Array = wsample(population, population_fitness_corrected, num_parents; replace=true, ordered=false)
    return parents
end

function rankBasedExpSelection(population::Array, population_fitness::Array{Float64}, num_parents::Int64, cfactor::Float64)::Array
    parent_ranks = sortperm(population_fitness; rev=true)
    parent_weights = map(x -> (1 - exp(-cfactor * x)), parent_ranks)
    parents::Array = wsample(population, parent_weights, num_parents; replace=true, ordered=false)

    return parents
end

function muGammaElitistSelection(population::Array, new_population::Array, new_fitnesses::Array{Float64}, old_fitnesses::Array{Float64}, elitism_ratio::Float64)
    population_cpy = deepcopy(population)
    number_of_elitists = Int32(floor(elitism_ratio*length(population_cpy)))

    # sort the population by the least fit
    ordered_indices = sortperm(old_fitnesses; rev=true)
    population_cpy[:] = population_cpy[ordered_indices]

    # sort the new population by the most fit
    ordered_indices = sortperm(new_fitnesses; rev=false)
    new_population[:] = new_population[ordered_indices]

    for i in 1:(length(population) - number_of_elitists)
        population_cpy[i] = new_population[i]
    end

    return population_cpy
end

end