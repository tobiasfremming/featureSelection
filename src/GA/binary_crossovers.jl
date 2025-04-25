module BinaryCrossovers

function onePointCrossover!(population::Vector{BitVector}, prob::Float64)
    """
    Applies standard one-point crossover within contiguous pairs of children. Splits two genotypes into two partitions and matches opposite partitions for every (contiguous) pair of children.
    Operations are in-place.

    Parameters:
        population::Vector{BitVector} the vector of children to crossover. Should have an even length.
        prob::Float64 the probability of applying the crossover.
    """
    for i in 1:Int32((floor(length(population)/2)))
        parent1 = population[2*i]
        parent2 = population[2*i-1]
        if rand() < prob
            crossover_point = Int32(floor(rand()*(length(population[1]) - 1)) + 1)
            for i in 1:crossover_point
                temp = parent1[i]
                parent1[i] = parent2[i]
                parent2[i] = temp
            end
        end
    end
end

function uniformCrossover!(population::Vector{BitVector}, prob::Float64)
    """
    Applies uniform crossover within contiguous pairs of children.
    
    Parameters:
        population::Vector{BitVector} the Vector of children to crossover. Should have an even length.
        prob::Float64 the probability of applying the crossover.
    """
    for i in 1:Int32((floor(length(population)/2)))
        if rand() < prob
            parent1 = population[2*i]
            parent2 = population[2*i-1]
            for i in 1:length(population[1])
                if rand() > 0.5
                    temp = parent1[i]
                    parent1[i] = parent2[i]
                    parent2[i] = temp
                end
            end     
        end
    end
end

end
