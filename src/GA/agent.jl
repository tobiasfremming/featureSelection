

module Agent


struct Expression
    from::Int64
    to::Int64
end


mutable struct Gene
    innovation::Int64
    expression::Expression
    weight::Float64 # weight of the expression. Between -1 and 1. Higher weights mean higher chance of activation for incoming bit
    bias::Float64 # threshhold for activation (bit getting turned on) for outgoing bit
    enabled::Bool
end


#  resetting the list every generation as opposed to keeping a growing list of mutations throughout evolution is sufficient to prevent an explosion of innovation numbers. 
global innovations = Dict{Expression, Int64}()


mutable struct Chromosome
    genes::Vector{Gene}
    fitness::Float64
    adjusted_fitness::Float64
end


mutable struct Species
    representative::Chromosome
    members::Vector{Chromosome}
end


function get_highest_and_lowest_innovation_number(chromosome::Chromosome)
    highest::Int64 = -1
    lowest::Int64 = 100000
    for gene::Gene in chromosome.genes
        if (gene.innovation > highest)
            highest = gene.innovation
        end
        if (gene.innovation < lowest)
            lowest = gene.innovation
        end
    end
    return (highest, lowest)
end


function get_innovation_range(chromosome1::Chromosome, chromosome2::Chromosome)
    min1, max1 = get_highest_and_lowest_innovation_number(chromosome1)
    min2, max2 = get_highest_and_lowest_innovation_number(chromosome2)
    return (max(min1, min2), min(max1, max2))
end


function calculate_number_of_disjoint_and_excess_genes(
    min::Int64, 
    max::Int64, 
    chromosome1::Chromosome, 
    chromosome2::Chromosome
    )
    # Collect innovation numbers
    genes1 = Dict(gene.innovation => gene for gene in chromosome1.genes)
    genes2 = Dict(gene.innovation => gene for gene in chromosome2.genes)

    all_innovations = union(keys(genes1), keys(genes2))

    disjoint_genes = 0
    excess_genes = 0

    # TODO: Can this be parallelized?
    # TODO: calculate enable flag difference here instead of in genomic_distance
    for innovation in all_innovations
        in1 = haskey(genes1, innovation)
        in2 = haskey(genes2, innovation)

        if xor(in1, in2)  # gene only exists in one chromosome
            if innovation < min || innovation > max
                excess_genes += 1
            else
                disjoint_genes += 1
            end
        end
    end

    return (disjoint_genes, excess_genes)
end


function genomic_distance(
    chromosome1::Chromosome, 
    chromosome2::Chromosome, 
    c1::Float64, 
    c2::Float64, 
    c3::Float64
    )
    min_range, max_range = get_innovation_range(chromosome1, chromosome2)
    disjoint, excess = calculate_number_of_disjoint_and_excess_genes(min_range, max_range, chromosome1, chromosome2)

    # Calculate average disabled difference
    
    genes2 = Dict(gene.innovation => gene for gene in chromosome2.genes)


    # TODO: use weights instead of enabled flag, maybe run this in another loop.
    # for gene1 in chromosome1.genes
    #     if haskey(genes2, gene1.innovation)
    #         gene2 = genes2[gene1.innovation]
    #         difference = gene1.enabled != gene2.enabled
    #             disabled_difference += difference
    #             enabled_difference += !difference
                
            
    #     end
    # end
    weights = 0.0
    for gene1 in chromosome1.genes
        if haskey(genes2, gene1.innovation)
            gene2 = genes2[gene1.innovation]
            
            weights += (gene1.weight - gene2.weight)^2 + (gene1.bias - gene2.bias)^2
        end
    end
    
    #enabled_flag_diff_ratio::Float64 = Float64(disabled_difference) / Float64(enabled_difference + 1e-7)
    N = max(min(length(chromosome1.genes), length(chromosome2.genes)), 1)
    return c1 * excess / N + c2 * disjoint / N + c3 * weights # add /N at the end there?

end


function fitness_sharing!(
    population::Vector{Chromosome}, 
    c1::Float64, 
    c2::Float64, 
    c3::Float64, 
    delta_t::Float64
    )
    Threads.@threads for i in 1:length(population)
        s = 0.0
        @inbounds for j in 1:length(population)
            δ = genomic_distance(population[i], population[j], c1, c2, c3)
            if δ < delta_t
                s += 1.0 - (δ / delta_t)
            end
        end
        if s > 0
            population[i].adjusted_fitness = population[i].fitness / s
        else
            population[i].adjusted_fitness = population[i].fitness
        end
    end
end


function assign_species(
    population::Vector{Chromosome}, 
    c1::Float64, 
    c2::Float64, 
    c3::Float64, 
    delta_t::Float64
    )
    species_list = Species[]  # all species
    
    for chrom in population
        assigned = false

        for sp in species_list
            δ = genomic_distance(chrom, sp.representative, c1, c2, c3)
            if δ < delta_t
                push!(sp.members, chrom)
                assigned = true
                break
            end
        end

        if !assigned
            push!(species_list, Species(chrom, [chrom]))
        end
    end

    # Update representatives for each species
    for sp in species_list
        sp.representative = rand(sp.members)
    end

    return species_list
end


#TODO: Keep track of species stagnation: if a species doesn’t improve for N generations, kill it off.
# TODO: maybe: Track the best genome per species, for elitism and representative updates.


function speciate_and_fitness_sharing!(
    population::Vector{Chromosome}, 
    c1::Float64, 
    c2::Float64, 
    c3::Float64, 
    delta_t::Float64
    )
    species_list = assign_species(population, c1, c2, c3, delta_t)
    fitness_sharing!(population, c1, c2, c3, delta_t)
    return species_list
end


function create_random_chromosome(num_genes::Int64, from::Int64, to::Int64)
    genes = Gene[]
    for i in 1:num_genes
        if rand(Bool) # 50% chance of creating a gene
            continue
        end
        gene = create_gene(true, from, to) 
        push!(genes, gene)
    end
    return Chromosome(genes, 0.0, 0.0)
end


function create_expression_from_start(start::Int64, stop::Int64)
    to = rand(start:stop)
    return Expression(start, to)
end


function create_random_expression(start::Int, stop::Int)
    if start > stop
        start, stop = stop, start  # swap to make a valid range
    end
    from = rand(start:stop)
    to = rand(start:stop)
    return Expression(from, to)
end


function create_gene(enabled::Bool, from::Int64, to::Int64)
    expression = create_random_expression(from, to)
    # weight and bias are random between -1 and 1
    weight = 2 * rand() - 1
    bias = 2 * rand() - 1
    if (haskey(innovations, expression))
        innovation = innovations[expression]
    else
        innovation = length(innovations) + 1
        innovations[expression] = innovation
    end

    return Gene(innovation, expression, weight, bias, enabled)
end




end





