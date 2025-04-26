module BinaryMutators

function applyMutationStandard!(population_to_mutate::Array{BitVector}, bitwiseProb::Float64)
    """
    Applies standard mutation

    Parameters:
        population_to_mutate::Array{BitVector} the population to apply mutation to
        bitwiseProb::Float64 the bitwise probability of mutation being applied
    """
    person_length = length(population_to_mutate[1])
    for person in population_to_mutate
        old_person = deepcopy(person)
        flipmask = rand(person_length) .< bitwiseProb
        person[flipmask] .= .!person[flipmask]
        if count(person) == 0
            person .= deepcopy(old_person)
        end
    end
end

end