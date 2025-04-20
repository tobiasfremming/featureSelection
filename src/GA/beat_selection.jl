
module BeatSelection

include("agent.jl")
using .Agent

function select_parent(population::Vector{Agent.Chromosome})
    total_fitness = sum(chromo.adjusted_fitness for chromo in population)
    pick = rand() * total_fitness
    current = 0.0
    for chromo in population
        current += chromo.adjusted_fitness
        if current > pick
            return chromo
        end
    end
    return population[end]  # fallback
end

end