module Mutations

using Random

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

function mutate!(chromosome::Chromosome; prob_weight=0.8, prob_toggle=0.1, prob_add_gene=0.05)
    for gene in chromosome.genes
        if rand() < prob_weight
            gene.weight += 0.1 * (2 * rand() - 1)  # small perturbation
            gene.weight = clamp(gene.weight, -1, 1)
        end

        if rand() < prob_toggle
            gene.enabled = !gene.enabled
        end
    end

    if rand() < prob_add_gene
        # Randomly insert a new gene
        new_gene = Agent.create_gene(true, 0, 10)  # You may want to pass limits here
        push!(chromosome.genes, new_gene)
    end
end


end