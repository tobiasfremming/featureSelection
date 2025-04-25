module Initialisers

using Random

function initialisePopulation(nsize::Int, gene_size::Int)::Vector{BitVector}
    """
    Creates a random and uniform population

    Parameters:
        nsize::Int the size of the population
    """
    population::Vector{BitVector} = Vector{BitVector}(undef, nsize)
    for i in 1:nsize
        person = bitrand(gene_size)
        population[i] = person
    end

    return population
end

end