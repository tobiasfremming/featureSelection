using Random, Statistics

# Sigmoid function to convert velocity to probability
sigmoid(x) = 1 / (1 + exp(-x))

# Define fitness function using your lookup table
# This example uses a placeholder function
# function fitness(bitstring::Vector{Int})
#     # Replace this with your lookup table logic
#     # Example placeholder: maximize sum of selected features
#     return sum(bitstring)
# end


struct NKLandscape
    N::Int
    K::Int
    interaction_indices::Vector{Vector{Int}}
    contribution_table::Vector{Dict{BitVector,Float64}}
end

# fitness landscape



# Initialize NK-landscape
function NKLandscape(N::Int, K::Int)
    interaction_indices = [sort(randperm(N)[1:K]) for _ in 1:N]
    contribution_table = [Dict{BitVector, Float64}() for _ in 1:N]

    for i in 1:N
        num_entries = 2^(K+1)
        for j in 0:num_entries-1
            bits = bitstring(j)[end-K:end]
            key = BitVector(parse.(Int, collect(bits)))
            contribution_table[i][key] = rand()
        end
    end

    return NKLandscape(N, K, interaction_indices, contribution_table)
end

# Fitness calculation
function fitness2(nk::NKLandscape, bitstring::Vector{Int})
    fitness_sum = 0.0
    for i in 1:nk.N
        idxs = [i; nk.interaction_indices[i]]
        key = BitVector(bitstring[idxs])
        fitness_sum += nk.contribution_table[i][key]
    end
    return fitness_sum / nk.N  # Average contribution
end

function fitness(nk::NKLandscape, bitstring::Vector{Int})
    ones = 0
    zeros = 0
    for i in 1:length(bitstring)
        if bitstring[i] == 1
            ones += 0.4
        else
            zeros += 1
        end
    end
    return max(ones, zeros) / length(bitstring)
end

# Initialize PSO parameters
mutable struct Particle
    X::Vector{Int}        # Position (bitstring)
    V::Vector{Float64}    # Velocity
    Pbest::Vector{Int}    # Personal best position
    Pbest_fitness::Float64
end

function init_swarm(num_particles::Int, num_features::Int)
    swarm = Particle[]
    for _ in 1:num_particles
        X = rand(0:1, num_features)
        V = randn(num_features)
        Pbest = copy(X)
        Pbest_fitness = fitness(X)
        push!(swarm, Particle(X, V, Pbest, Pbest_fitness))
    end
    return swarm
end

function binary_pso(fitness; num_features::Int, num_particles::Int=50, iterations::Int=50, nk::NKLandscape)
    swarm = init_swarm(num_particles, num_features)
    w_start, w_end = 0.9, 0.4  # Dynamically decreasing inertia
    c1, c2 = 2.2, 2.5          # Slightly favor social component
    particle_stagnation = zeros(Int, num_particles)
    stagnation_limit = 15
    V_max = 6.0  


    # Initialize global best
    Gbest = copy(swarm[1].Pbest)
    Gbest_fitness = swarm[1].Pbest_fitness
    for p in swarm
        if p.Pbest_fitness > Gbest_fitness
            Gbest = copy(p.Pbest)
            Gbest_fitness = p.Pbest_fitness
        end
    end

    # Main loop
    for iter in 1:iterations
        w = w_start - (w_start - w_end)*(iter/iterations)

        for (i, p) in enumerate(swarm)

            current_fitness = fitness(nk, p.X)
            
            if current_fitness > p.Pbest_fitness
                p.Pbest = copy(p.X)
                p.Pbest_fitness = current_fitness
                particle_stagnation[i] = 0
                
                if current_fitness > Gbest_fitness
                    Gbest = copy(p.X)
                    Gbest_fitness = current_fitness
                end
            else
                particle_stagnation[i] += 1
            end

            


            for j in 1:num_features
                # Update velocity
                r1, r2 = rand(), rand()
                cognitive = c1 * r1 * (p.Pbest[j] - p.X[j])
                social = c2 * r2 * (Gbest[j] - p.X[j])
                # p.V[j] = w * p.V[j] + cognitive + social
                # clamp velocity
                p.V[j] = clamp(w * p.V[j] + cognitive + social, -V_max, V_max)


                # Update position (binary)
                prob = sigmoid(p.V[j])
                p.X[j] = rand() < prob ? 1 : 0
            end

            mutation_rate = 0.05  # 2% chance mutation per bit

            for p in swarm
                if rand() < mutation_rate
                    idx = rand(1:num_features)
                    p.X[idx] = 1 - p.X[idx]  # flip bit
                end
            end

            # Evaluate fitness
            current_fitness = fitness(nk, p.X)

            # Update personal best
            if current_fitness > p.Pbest_fitness
                p.Pbest = copy(p.X)
                p.Pbest_fitness = current_fitness

                # Update global best
                if current_fitness > Gbest_fitness
                    Gbest = copy(p.X)
                    Gbest_fitness = current_fitness
                end
            end
        end

        # (Optional) Track progress
        println("Iteration $iter, Best fitness: $Gbest_fitness")
    end

    return Gbest, Gbest_fitness
end

# Example Usage:
num_features = 100  # Replace with actual number of features
K = 5   # Number of interactions per feature (higher = harder)

nk = NKLandscape(num_features, K)
best_subset, best_fitness = binary_pso(fitness; num_features=num_features, iterations=1000, nk)

println("Best subset found: ", best_subset)
println("Best subset fitness: ", best_fitness)
