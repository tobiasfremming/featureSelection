




function crossover2(chromosome1::Agent.Chromosome, chromosome2::Agent.Chromosome)
    offspring_genes = Vector{Agent.Gene}()
    genes1 = chromosome1.genes
    genes2 = chromosome2.genes
    #(max_innovation, min_innovation) = AgentModule.get_innovation_range(chromosome1, chromosome2)
    #(disjoint_genes, excess_genes) = AgentModule.calculate_number_of_disjoint_and_excess_genes(min_innovation, max_innovation, chromosome1, chromosome2)
    genes1_dict = Dict(gene.innovation => gene for gene in genes1)
    genes2_dict = Dict(gene.innovation => gene for gene in genes2)
    all_innovations = union(keys(genes1_dict), keys(genes2_dict))

    for innovation in all_innovations
        gene1 = get(genes1_dict, innovation, nothing)
        gene2 = get(genes2_dict, innovation, nothing)

        if gene1 === nothing
                push!(offspring_genes, deepcopy(gene2))
        elseif gene2 === nothing
                push!(offspring_genes, deepcopy(gene1))
        else
            if rand() < 0.5
                push!(offspring_genes, deepcopy(gene1))
            else
                push!(offspring_genes, deepcopy(gene2))
            end
        end
    end

    return Agent.Chromosome(offspring_genes, 0.0, 0.0)

    
end


function crossover(parent1::Agent.Chromosome, parent2::Agent.Chromosome)
    # Ensure parent1 is more fit or randomly pick if equal
    if parent2.fitness > parent1.fitness
        parent1, parent2 = parent2, parent1
    end

    genes1 = Dict(g.innovation => g for g in parent1.genes)
    genes2 = Dict(g.innovation => g for g in parent2.genes)

    all_innovations = union(keys(genes1), keys(genes2))
    offspring_genes = Agent.Gene[]

    for innov in sort(collect(all_innovations))
        g1 = get(genes1, innov, nothing)
        g2 = get(genes2, innov, nothing)

        if g1 !== nothing && g2 !== nothing
            # Matching gene: randomly choose one
            push!(offspring_genes, deepcopy(rand(Bool) ? g1 : g2))
        elseif g1 !== nothing
            # Disjoint or excess from more fit parent (parent1)
            push!(offspring_genes, deepcopy(g1))
        end
        # Note: disjoint/excess genes from less fit parent are skipped
    end

    return Agent.Chromosome(offspring_genes, 0.0, 0.0)
end










