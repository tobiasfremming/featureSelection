
include("agent.jl")
using .AgentModule

module Crossover


function crossover(chromosome1::Chromosome, chromosome2::Chromosome)
    offspring_genes = Vector{Gene}()
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


    
end












end # module
