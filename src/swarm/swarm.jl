using Random, Statistics

# Sigmoid function to convert velocity to probability
sigmoid(x) = 1 / (1 + exp(-x))

# Define fitness function using your lookup table
# This example uses a placeholder function
function fitness(bitstring::Vector{Int})
    # Replace this with your lookup table logic
    # Example placeholder: maximize sum of selected features
    return sum(bitstring)
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

function binary_pso(fitness; num_features::Int, num_particles::Int=50, iterations::Int=50)
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

            current_fitness = fitness(p.X)
            
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
            current_fitness = fitness(p.X)

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
best_subset, best_fitness = binary_pso(fitness; num_features=num_features, iterations=1000)

println("Best subset found: ", best_subset)
println("Best subset fitness: ", best_fitness)
