module local_search

using StatsBase

export local_bitflip_search, local_greedy_search

"""
    local_bitflip_search(individual, fitness, feature_impact, feature_counts)

Performs a local bitflip search on the individual. Evaluates the fitness impact
of flipping each bit, updates the global feature impact table, and returns the
best individual found along with updated impact and counts.

# Arguments
- `individual::Vector{Bool}`: Binary feature vector.
- `fitness::Function`: A function that returns the fitness score for an individual.
- `feature_impact::Vector{Float64}`: Global feature impact table (mutable).
- `feature_counts::Vector{Int}`: Global table of counts per feature (mutable).

# Returns
- `best_individual::Vector{Bool}`
- `updated_feature_impact::Vector{Float64}`
- `updated_feature_counts::Vector{Int}`
"""
function local_bitflip_search(individual::Vector{Bool},
                              fitness::Function,
                              lut::Dict{Vector{Bool}, Float64},
                              feature_impact::Vector{Float64},
                              feature_counts::Vector{Int})

    n = length(individual)
    base_fitness = fitness(individual, lut)
    best_individual = copy(individual)
    best_fitness = base_fitness

    for i in 1:3#n  # use just 3 local searches as test, change to n later
        # k = i
        k = rand(1:n)
        original_value = individual[k]
        neighbor = copy(individual)
        
        # Flip the k-th bit
        neighbor[k] = !neighbor[k]
        
        new_fitness = fitness(neighbor, lut)
        delta = base_fitness - new_fitness  # directional change in fitness
        
        # Update feature impact table: track whether it's a gain or loss in performance
        if original_value  # If the feature was ON, and we turned it OFF
            feature_impact[k] -= delta  
        else  # If the feature was OFF, and we turned it ON
            feature_impact[k] += delta 
        end

        feature_counts[k] += 1
        
        # Keep the best local variant
        if new_fitness < best_fitness
            best_individual = neighbor
            best_fitness = new_fitness
        end
    end

    return best_individual, feature_impact, feature_counts
end


function local_greedy_search(individual::Vector{Bool}, f::Float64, fitness::Function, lut::Dict{Vector{Bool}, Float64}, feature_impact::Vector{Float64}, feature_counts::Vector{Int}, n::Int)
    mutated_indiv = copy(individual)

    best_individual = mutated_indiv
    best_fitness = copy(f)
    
    on_features  = findall(x -> x, individual)
    off_features = findall(x -> !x, individual)
    
    # Rescale scores to [0, 1]
    feature_scores = feature_impact ./ (feature_counts .+ 1e-3)
    score_probs = (feature_scores .- minimum(feature_scores)) ./ (maximum(feature_scores) - minimum(feature_scores) + 1e-12)

    # worst_on  = sort(on_features,  by = i -> feature_scores[i])  # ascending (lowest scores first)
    # best_off = sort(off_features, by = i -> -feature_scores[i]) # descending (highest scores first)

    off_probs = score_probs[off_features]
    on_probs = 1 .- score_probs[on_features]
    # println(off_probs, on_probs)

    for i in 1:n

        if rand() < 0.5  # 50% chance to either flip worst bit OFF or best bit ON
            # For OFF features, higher score ⇒ more likely to flip ON
            if length(off_probs) > 0
                chosen_off = sample(off_features, Weights(off_probs))
                mutated_indiv[chosen_off] = !mutated_indiv[chosen_off]  # flip the bit
            end
        else
            # For ON features, lower score ⇒ more likely to flip OFF
            if length(on_probs) > 0
                chosen_on = sample(on_features, Weights(on_probs))
                mutated_indiv[chosen_on] = !mutated_indiv[chosen_on]  # flip the bit
            end
        end
        
        new_fitness = fitness(mutated_indiv, lut)

        # Keep the best local variant
        if new_fitness < best_fitness
            best_individual = mutated_indiv
            best_fitness = new_fitness
        end
    end

    if best_fitness < f
        return best_individual, best_fitness
    else
        return individual, f
    end
    
end


end # end module