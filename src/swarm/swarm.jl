using Random, Statistics

using CSV, DataFrames                      

const DATASET = get(ENV, "DATASET", "diabetes")   # choose set via ENV var

# File is in   src/Visualization_and_SGA/Lookup_tables/
const LUT_PATH = joinpath(@__DIR__, "..","..", "Visualization_and_SGA",
                          "Lookup_tables", "$(DATASET)_fitness_lut.csv")

@assert isfile(LUT_PATH) "Lookup-table not found:\n$LUT_PATH"

const lut_df = CSV.read(LUT_PATH, DataFrame;
                        types = Dict(1=>String, 2=>Float64, 3=>Float64,
                                     4=>Float64, 5=>Float64, 6=>Float64))

# number of bits = width of the *string* in column 1
const GENE_SIZE = length(string(lut_df.key[end]))

# turn the Int / String keys into BitVector so we can index with the
# particle bit-string directly
lut_df.key = lpad.(string.(lut_df.key), GENE_SIZE, '0')
lut_df.key = [BitVector(c=='1' for c in s) for s in lut_df.key]

# one dictionary: BitVector → fitness (column f16 is what SGA minimises)
const LUT_FITNESS = Dict(lut_df.key .=> lut_df.f16)

# If you also need the “err” column later you can build
# const LUT_ERROR = Dict(lut_df.key .=> lut_df.err)
# ──────────────────────────────────────────────────────────────────
#  FITNESS   (SGA minimised ⇒ PSO must MAXIMISE −value)  ───────────
# ──────────────────────────────────────────────────────────────────
@inline function fitness(bits::Vector{Int})::Float64
    return -LUT_FITNESS[bits]      # negate because PSO maximises
end
# ──────────────────────────────────────────────────────────────────


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

function fitness3(bitstring::Vector{Int})
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

using Random


# Main -----------------------------------------------------------------
function binary_pso(fitness;        # a function that takes a bit-vector
                    num_features::Int,
                    num_particles::Int = 50,
                    iterations::Int  = 50,
                    nk = nothing)    # kept only so the call-site doesn’t change

    swarm = init_swarm(num_particles, num_features)

    w_start, w_end = 0.9, 0.4        # inertia schedule
    c1, c2          = 2.2, 2.5       # cognitive / social factors
    V_max           = 6.0

    particle_stagnation = zeros(Int, num_particles)
    stagnation_limit    = 15         # per-particle limit

    # global best -------------------------------------------------------
    Gbest          = copy(swarm[1].Pbest)
    Gbest_fitness  = swarm[1].Pbest_fitness

    for p in swarm
        if p.Pbest_fitness > Gbest_fitness
            Gbest, Gbest_fitness = copy(p.Pbest), p.Pbest_fitness
        end
    end

    # swarm-level stagnation tracking ----------------------------------
    last_gbest           = Gbest_fitness
    swarm_stagnation     = 0
    swarm_stagn_limit    = 25        # iterations

    # optimisation loop -------------------------------------------------
    for iter in 1:iterations
        w = w_start - (w_start - w_end) * (iter / iterations)

        for (i, p) in enumerate(swarm)

            # ------- velocity & position update (binary) --------------
            for j in 1:num_features
                r1, r2  = rand(), rand()
                cognitive = c1 * r1 * (p.Pbest[j] - p.X[j])
                social    = c2 * r2 * (Gbest[j]  - p.X[j])

                p.V[j] = clamp(w * p.V[j] + cognitive + social, -V_max, V_max)
                p.X[j] = rand() < sigmoid(p.V[j]) ? 1 : 0
            end

            # ------- fitness / personal best --------------------------
            current_fitness = fitness(p.X)

            if current_fitness > p.Pbest_fitness
                p.Pbest, p.Pbest_fitness = copy(p.X), current_fitness
                particle_stagnation[i]    = 0
            else
                particle_stagnation[i]   += 1
                if particle_stagnation[i] ≥ stagnation_limit
                    # kick velocity
                    p.V .= randn(num_features) .* V_max
                    # flip 10 % random bits
                    nflip      = max(1, round(Int, 0.10 * num_features))
                    flip_index = rand(1:num_features, nflip)
                    p.X[flip_index] .= 1 .- p.X[flip_index]
                    particle_stagnation[i] = 0
                end
            end

            # ------- global best --------------------------------------
            if p.Pbest_fitness > Gbest_fitness
                Gbest, Gbest_fitness = copy(p.Pbest), p.Pbest_fitness
            end
        end  # end particle loop

        # ------- global mutation pass ---------------------------------
        # mutation_rate = 0.05
        # for p in swarm
        #     if rand() < mutation_rate
        #         idx = rand(1:num_features)
        #         p.X[idx] = 1 - p.X[idx]
        #     end
        # end

        # ------- swarm-level stagnation check -------------------------
        if Gbest_fitness == last_gbest
            swarm_stagnation += 1
        else
            swarm_stagnation = 0
            last_gbest       = Gbest_fitness
        end

        if swarm_stagnation ≥ swarm_stagn_limit
            # reseed the worst half of the swarm
            worst = sortperm([p.Pbest_fitness for p in swarm])[1:div(num_particles, 2)]
            for idx in worst
                swarm[idx].X .= rand(0:1, num_features)
                swarm[idx].V .= randn(num_features)
                swarm[idx].Pbest = copy(swarm[idx].X)
                swarm[idx].Pbest_fitness = fitness(swarm[idx].X)
                particle_stagnation[idx] = 0
            end
            swarm_stagnation = 0
        end

        println("Iteration $iter | best fitness = $Gbest_fitness")
    end

    return Gbest, Gbest_fitness
end


# num_features = 100
K            = GENE_SIZE
nk  = NKLandscape(GENE_SIZE, K)   # still built even if not used inside PSO
best_subset, best_fit = binary_pso(fitness; num_features=GENE_SIZE,
                                   iterations=1000, nk=nk)
println("Best subset:  ", best_subset)
println("Best fitness: ", best_fit)


