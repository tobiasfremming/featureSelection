
using Random
using StatsBase
# using Plots
# using Dates
# using FileIO
# using DataFrames
# using CSV





function get_population_ranks(fitnesses::Vector{Float64})
    N = length(fitnesses)
    ranks = fill(1, N)

    for i in 1:N
        for j in 1:N
            if i != j
                if fitnesses[i] < fitnesses[j]  # lower is better (error)
                    ranks[j] = max(ranks[j], ranks[i] + 1)
                elseif fitnesses[i] > fitnesses[j]
                    ranks[i] = max(ranks[i], ranks[j] + 1)
                end
            end
        end
    end

    return ranks
end

function get_population_crowding(fitnesses::Vector{Float64}, ranks::Vector{Int})
    crowdings = fill(0.0, length(fitnesses))

    for rank in 1:maximum(ranks)
        indices = findall(x -> x == rank, ranks)
        if length(indices) < 3
            for idx in indices
                crowdings[idx] = Inf
            end
            continue
        end

        sorted = sortperm(fitnesses[indices], rev=true)
        f_sorted = fitnesses[indices][sorted]
        idx_sorted = indices[sorted]

        min_f = f_sorted[end]
        max_f = f_sorted[1]

        crowdings[idx_sorted[1]] = Inf
        crowdings[idx_sorted[end]] = Inf

        for i in 2:length(f_sorted)-1
            crowdings[idx_sorted[i]] = (f_sorted[i - 1] - f_sorted[i + 1]) / (max_f - min_f + 1e-9)
        end
    end

    return crowdings
end

function select_parents_nsga(population::Vector{Agent.Chromosome}, nparents::Int)::Vector{Agent.Chromosome}
    fitnesses = [c.fitness for c in population]
    ranks = get_population_ranks(fitnesses)
    crowdings = get_population_crowding(fitnesses, ranks)

    selected = Agent.Chromosome[]
    for _ in 1:nparents
        if length(population) < 2
            push!(selected, deepcopy(population[1]))  # fallback
            continue
        end
    
        i, j = sample(1:length(population), 2, replace=false)
    
        if ranks[i] < ranks[j]
            push!(selected, deepcopy(population[i]))
        elseif ranks[j] < ranks[i]
            push!(selected, deepcopy(population[j]))
        else
            chosen = crowdings[i] > crowdings[j] ? i : j
            push!(selected, deepcopy(population[chosen]))
        end
    end
    

    return selected
end

