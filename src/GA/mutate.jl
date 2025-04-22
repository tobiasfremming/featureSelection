module Mutations

using ..Agent

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



function mutate!(
    chromosome::Agent.Chromosome;
    prob_weight_perturb=0.8,
    prob_weight_reset=0.1,
    prob_toggle=0.01,
    prob_add_gene=0.05,
    prob_add_node=0.03,
    max_node_id=100  # assume we know the node ID space
)
    # === 1. Weight mutation ===
    for gene in chromosome.genes
        if rand() < prob_weight_perturb
            gene.weight += 0.1 * (2 * rand() - 1)  # small change
            gene.weight = clamp(gene.weight, -1, 1)
        elseif rand() < prob_weight_reset
            gene.weight = 2 * rand() - 1
        end
    end

    # === 2. Enable/disable toggling ===
    for gene in chromosome.genes
        if rand() < prob_toggle
            gene.enabled = !gene.enabled
        end
    end

    # === 3. Add new connection ===
    if rand() < prob_add_gene
        from_node = rand(0:max_node_id)
        to_node = rand(0:max_node_id)
        while from_node == to_node  # avoid self-loop
            to_node = rand(0:max_node_id)
        end
        new_gene = Agent.create_gene(true, from_node, to_node)
        push!(chromosome.genes, new_gene)
    end

    # === 4. Add node (split connection) ===
    if rand() < prob_add_node && !isempty(chromosome.genes)
        # pick a random enabled gene to split
        candidates = filter(g -> g.enabled, chromosome.genes)
        if !isempty(candidates)
            gene = rand(candidates)
            gene.enabled = false  # disable old gene

            new_node = rand(max_node_id+1:max_node_id+10)  # create a new node ID
            # Gene from original input to new node
            gene1 = Agent.create_gene(true, gene.expression.from, new_node)
            gene1.weight = 1.0  # optional: direct transfer

            # Gene from new node to original output
            gene2 = Agent.create_gene(true, new_node, gene.expression.to)
            gene2.weight = gene.weight  # preserve original influence

            push!(chromosome.genes, gene1)
            push!(chromosome.genes, gene2)
        end
    end
end



end