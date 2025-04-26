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
        while true
            person = bitrand(gene_size)
            
            if count(person) != 0
                population[i] = person
                break
            end
        end
    end

    return population
end

end