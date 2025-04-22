using CSV
using DataFrames
using Random
using StatsBase
using Statistics
using Plots
using Printf
using Base.Threads

include("Local_search.jl")
using .local_search 

lut_cancer = joinpath(@__DIR__, "Lookup_tables/cancer_fitness_lut.csv")
lut_diabetes = joinpath(@__DIR__, "Lookup_tables/diabetes_fitness_lut.csv")
lut_heart = joinpath(@__DIR__, "Lookup_tables/heart_fitness_lut.csv")
lut_df = CSV.read(lut_heart,DataFrame,header=1,types=Dict(1 => String, 2 => Float64, 3 => Float64, 4 => Float64, 5 => Float64, 6 => Float64))
lut_int = Dict(lut_df.key .=> lut_df.f16)
GENE_SIZE = length(string(lut_df[end,1]))

# Convert Column1 to Vector{Bool}
lut_df.key = [lpad(string(x), GENE_SIZE, '0') for x in lut_df.key]  # converts Ints to padded strings
lut_df.key = [[c == '1' for c in s] for s in lut_df.key]  # converts strings to Vector{Bool}
lut = Dict(lut_df.key .=> lut_df.f16)
lut_model_error = Dict(lut_df.key .=> lut_df.err)


# random population generator
function pop_generator(n::Int, length::Int=GENE_SIZE)
    return [rand(Bool, length) for _ in 1:n]
    # return rand(Bool, n, length)
end
test_pop = pop_generator(5)

# fitness function for bool_vectors
function fitness(individual::Vector{Bool}, lut::Dict)
    # in case of incomplete lut, calculate fitness from ML model:
    # X_sub = get_columns(X, bitstring)
    # return -get_fitness(model, X_sub, y; rng=StableRNG(12))
    return lut[individual]
end

# fitness function for Ints
function fitness(individual::Int, lut::Dict)
    # in case of incomplete lut, calculate fitness from ML model:
    # X_sub = get_columns(X, bitstring)
    # return -get_fitness(model, X_sub, y; rng=StableRNG(12))
    return lut[individual]
end

# fitness function for population of matrix
function fitness(population::Matrix, lookup_table::Dict)
    # in case of incomplete lut, calculate fitness from ML model:
    # X_sub = get_columns(X, bitstring)
    # return -get_fitness(model, X_sub, y; rng=StableRNG(12))
    fitness_values = map(i -> fitness(vec(population[i, :]), lookup_table), 1:size(population, 1))
    return fitness_values
end

# fitness function for population of vector{vector{Bool}}
function fitness(population::Vector{Vector{Bool}}, lookup_table::Dict)
    fitness_values = Vector{Float64}(undef, length(population))
    Threads.@threads for i in eachindex(population)
        fitness_values[i] = fitness(population[i], lookup_table)
    end
    # fitness_values = map(x -> fitness(x, lookup_table), population)
    return fitness_values
end

f = fitness(test_pop, lut)
test::Vector{Bool} = Bool[0,  1,  0,  1,  1,  1,  0 , 1]
f = lut[test]
f = fitness(test_pop, lut)

# parent selector - rank based
function parent_selector(fitness::Vector{Float64}, λ=0.8)
    f = -fitness  # invert to minimize
    num_parents = length(f)
    # Sort fitness values in descending order (higher is better)
    sorted_indices = sortperm(f, rev=true)  # Higher fitness = better rank
    ranks = invperm(sorted_indices)  # Convert sorted order into ranks (best = 1, worst = N)
    
    # Compute exponential probabilities based on rank
    rank_probabilities = exp.(-λ .* ranks)  # Exponential weighting
    rank_probabilities ./= sum(rank_probabilities)  # Normalize to sum to 1

    # Select parents using weighted sampling
    selected_indices = sample(1:length(f), Weights(rank_probabilities), num_parents, replace=true)

    return selected_indices
end
parents = parent_selector(f)
parents = test_pop[parents, :]


# crossover method for 2 parents 
function one_point_crossover(mom::Vector{Bool}, dad::Vector{Bool}, p::Float64=0.8)
    if rand() > p
        return mom, dad
    else
        split_idx = Int(round(rand() * length(mom)))
        # midpoint = div(length(mom), 2)
        mom_left, mom_right = mom[1:split_idx], mom[split_idx+1:end]
        dad_left, dad_right = dad[1:split_idx], dad[split_idx+1:end]
        child1 = vcat(mom_left, dad_right)
        child2 = vcat(dad_left, mom_right)
        return child1, child2
    end
end


# crossover method for the entire population of parents::Vector{Vector{Bool}}
function one_point_crossover(pop::Vector{Vector{Bool}}, p::Float64 = 0.8)
    parents = shuffle(pop)
    children = Vector{Vector{Bool}}(undef, length(parents))  # initialize empty children
    for i in 1:2:length(parents)-1
        children[i], children[i+1] = one_point_crossover(parents[i], parents[i+1], p)
    end
    if length(parents) % 2 == 1  # in case there are odd number of individuals
        children[end] = parents[rand(1:pop_size)]  # copy a random parent to fill up the children
    end
    return children
end


# # crossover method for entire population of parents
# function one_point_crossover(pop::Matrix{Bool}, p::Float64 = 0.8)
#     num_individuals, chromosome_length = size(pop)

#     # Shuffle the population indices for random pairing
#     shuffled_indices = shuffle(1:num_individuals)
#     shuffled_pop = pop[shuffled_indices, :]

#     # Container for offspring
#     offspring = similar(pop)

#     i = 1
#     while i <= num_individuals - 1
#         mom = shuffled_pop[i, :]
#         dad = shuffled_pop[i + 1, :]

#         child1, child2 = one_point_crossover(mom, dad, p)

#         offspring[i, :] = child1
#         offspring[i + 1, :] = child2
#         i += 2
#     end

#     # If there's an odd one out, copy them directly
#     if isodd(num_individuals)
#         offspring[end, :] = shuffled_pop[end, :]
#     end

#     return offspring
# end


# mutation
function mutation(bitstring::Vector{Bool}, p::Float64=0.005)
    mutation_mask = rand(length(bitstring)) .< p
    return bitstring .⊻ mutation_mask  # elementwise XOR operator flips bits if mask is true 
end

# mutate population::Vector{Vector{Bool}} in place
function mutate_pop!(pop::Vector{Vector{Bool}}, p::Float64=0.005)
    for i = 1:length(pop)
        pop[i] = mutation(pop[i], p)
    end    
end

# mutation
function mutate_population(pop::Matrix{Bool}, p::Float64=0.005)
    mutation_mask = rand(size(pop)) .< p  # mask of pop_size
    mutated_pop = pop .⊻ mutation_mask  # elementwise XOR operator flips bits if mask is true
    return Matrix{Bool}(mutated_pop)
end


function survivor_selection_f(pop_old::Vector{Vector{Bool}}, pop_new::Vector{Vector{Bool}}, 
    f_old::Vector{Float64}, f_new::Vector{Float64})
    pop_size = length(pop_old)

    # Calculate number of survivors from each
    num_old_survivors = round(Int, 0.2 * pop_size)
    num_new_survivors = pop_size - num_old_survivors

    # Sort by fitness (lower is better)
    old_sorted_indices = sortperm(f_old)
    new_sorted_indices = sortperm(f_new)

    # Select top individuals and their fitness values
    survivors_old = pop_old[old_sorted_indices[1:num_old_survivors]]
    survivors_old_fitness = f_old[old_sorted_indices[1:num_old_survivors]]

    survivors_new = pop_new[new_sorted_indices[1:num_new_survivors]]
    survivors_new_fitness = f_new[new_sorted_indices[1:num_new_survivors]]

    # Combine
    survivors = vcat(survivors_old, survivors_new)
    survivor_fitness = vcat(survivors_old_fitness, survivors_new_fitness)

    return survivors, survivor_fitness
end


# function survivor_selection_f(pop_old::Matrix{Bool}, pop_new::Matrix{Bool}, 
#                               f_old::Vector{Float64}, f_new::Vector{Float64})
#     pop_size = size(pop_old, 1)

#     # Calculate number of survivors from each
#     num_old_survivors = round(Int, 0.2 * pop_size)
#     num_new_survivors = pop_size - num_old_survivors

#     # Sort by fitness (lower is better)
#     old_sorted_indices = sortperm(f_old)
#     new_sorted_indices = sortperm(f_new)

#     # Select top individuals and their fitness values
#     survivors_old = pop_old[old_sorted_indices[1:num_old_survivors], :]
#     survivors_old_fitness = f_old[old_sorted_indices[1:num_old_survivors]]

#     survivors_new = pop_new[new_sorted_indices[1:num_new_survivors], :]
#     survivors_new_fitness = f_new[new_sorted_indices[1:num_new_survivors]]

#     # Combine
#     survivors = vcat(survivors_old, survivors_new)
#     survivor_fitness = vcat(survivors_old_fitness, survivors_new_fitness)

#     return survivors, survivor_fitness
# end


# parameter selection:
pop_size = 50
n_epochs = 100
p_crossover = 0.8
p_mutation = 0.03

function main()
    feature_impact = zeros(GENE_SIZE)
    feature_counts = zeros(Int, GENE_SIZE)
    # statistical tracking analysis
    epoch_mean = Vector{Float64}(undef, n_epochs+1)
    epoch_max = Vector{Float64}(undef, n_epochs+1)
    epoch_min = Vector{Float64}(undef, n_epochs+1)
    epoch_fittest_individual = Vector{Vector{Bool}}(undef, n_epochs+1)
    min_f = Inf
    best_indiv = Vector{Bool}(undef, 500)

    pop = pop_generator(pop_size)  # initialize a random population
    f = fitness(pop, lut)  # compute the fitness of the first population
    # println(length(pop))
    for e = 1:n_epochs
        min, idx = findmin(f)
        epoch_fittest_individual[e] = pop[idx]
        if min < min_f
            min_f = min
            best_indiv = pop[idx]
        end
        epoch_mean[e] = mean(f)
        epoch_max[e] = maximum(f) 
        epoch_min[e] = minimum(f)
        # diversity[e] = calc_diversity(pop)
        # println("Mean fitness: ", epoch_mean[e], " | ", "Max fitness: ", epoch_max[e])
        println("Epoch $e  |  Mean fitness: ", epoch_mean[e], " | ", "Min fitness: ", epoch_min[e])
        # parents = pop[tournament_selection(f)]
        parents = pop[parent_selector(f)]  # select the parents 
        # parents = shuffle(parents)  # shuffle the order for reproduction
        children = one_point_crossover(parents, p_crossover)
        mutate_pop!(children, p_mutation)
        f_children = fitness(children, lut)

        # create the feature_impact table and the feature_counts table
        # for i in eachindex(children)
        #     child = copy(children[i])
        #     best_local_solution, feature_impact, feature_counts = local_search.local_bitflip_search(child, fitness, lut_model_error, feature_impact, feature_counts)
        # end

        # # greedy local search on the children using the feature_impact table calculated above
        # for i in eachindex(children)
        #     child = copy(children[i])
        #     f_child = copy(f_children[i])
        #     best_local_indiv, its_fitness = local_search.local_greedy_search(child, f_child, fitness, lut, feature_impact, feature_counts, 4)
        #     children[i] = best_local_indiv
        #     f_children[i] = its_fitness
        #     # println(f_children[i])
        # end

        # pop = survivor_selection_f(pop, children)  # use fitness based survivior selection
        # pop = survivor_selection_g(pop, children)  # use generational survivior selection
        pop, f = survivor_selection_f(pop, children, f, f_children)
        e = e + 1
    end
    min, idx = findmin(f)
    if min < min_f
        min_f = min
        best_indiv = pop[idx]
    end
    epoch_mean[n_epochs+1] = mean(f)
    epoch_max[n_epochs+1] = maximum(f)
    epoch_min[n_epochs+1] = minimum(f)
    epoch_fittest_individual[n_epochs+1] = pop[idx]
    println("best indiv: ", best_indiv)

    plot(0:length(epoch_mean)-1, epoch_mean, label="Mean fitness", color=:blue, lw=2)  # First vector
    plot!(0:length(epoch_max)-1, epoch_max, label="Max fitness", color=:green, lw=2)  # Second vector (use plot! to overlay)
    plot!(0:length(epoch_min)-1, epoch_min, label="Min fitness", color=:red, lw=2)
    # plot(0:length(diversity)-1, diversity, label="diversity", color=:black, lw=2)
    return epoch_fittest_individual, epoch_min[end], feature_impact, feature_counts
end

epoch_fittest_individuals, f, feature_impact, feature_counts = main()
feature_impact ./ (feature_counts .+ 1e-3)
println("global minimum = ", lut_df[!,1][findmin(lut_df[!,5])[2]], minimum(lut_df[!,5]))

function statistical_analysis()
    n_runs = 100
    # Stats
    fitness = Vector{Float64}(undef, n_runs)
    best_indiv = Vector{Vector{Bool}}(undef, n_runs)
    num_iterations_to_reach_best_fitness = Vector{Int64}(undef, n_runs)
    for i in 1:n_runs
        solutions, f = main()
        fitness[i] = f
        best_indiv[i] = solutions[end]
        num_iterations_to_reach_best_fitness[i] = findfirst(x -> x == best_indiv[i], solutions)  # finds first appearance of best indiv
    end
    return fitness, best_indiv, num_iterations_to_reach_best_fitness
end

fitness_values, best_indivs, num_iterations_to_reach_best_fitness = statistical_analysis()
mean_f = mean(fitness_values)
std_f = std(fitness_values)
global_optimum = lut_df[!,1][findmin(lut_df[!,5])[2]]
best_possible_fitness = minimum(lut_df[!,5])
successful_runs = map(x -> x == global_optimum, best_indivs)
num_successful_runs = sum(map(x -> x == best_possible_fitness, fitness_values))
num_ones = count(==(1), successful_runs)
num_zeros = count(==(0), successful_runs)
mean_convergance_speed = mean(num_iterations_to_reach_best_fitness)
std_convergance_speed = std(num_iterations_to_reach_best_fitness)


f = map(x -> fitness(x, lut), epoch_fittest_individuals)
indiv = Bool[1, 1, 0, 1, 0, 1, 0, 1]
f = fitness(indiv, lut)



function int_to_bitvec(x::Int, nbits::Int=8)
    bin_str = lpad(string(x), nbits, '0')
    return BitVector(c == '1' for c in bin_str)
end