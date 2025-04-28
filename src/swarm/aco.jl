using Random, Statistics

# Fitness function (replace with your lookup logic)
# function fitness(bitstring::Vector{Int})
#     return sum(bitstring)
# end

using CSV, DataFrames                        # add these two

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
    return - LUT_FITNESS[bits]      # negate because PSO maximises
end
# ──────────────────────────────────────────────────────────────────


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

function fitness2(nk::NKLandscape, bitstring::Vector{Int})
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

function binary_aco(fitness; num_features=50, num_ants=50, iterations=100, α=1.0, ρ=0.15, Q=1.0, nk::NKLandscape)
   
    # Initialize pheromone trails
    pheromone = fill(0.5, num_features)

    best_solution = zeros(Int, num_features)
    best_fitness = -Inf

    for iter in 1:iterations
        ant_solutions = []
        ant_fitnesses = []

        for ant in 1:num_ants
            # Ant constructs a solution based on pheromone
            subset = [rand() < (pheromone[j]^α) / ((pheromone[j]^α) + (1 - pheromone[j])^α) ? 1 : 0 for j in 1:num_features]

            f = fitness(subset)
            push!(ant_solutions, subset)
            push!(ant_fitnesses, f)

            if f > best_fitness
                best_solution = copy(subset)
                best_fitness = f
            end
        end

        # Evaporation
        pheromone .= (1 - ρ) .* pheromone

        # Reinforcement (only best ant solution)
        best_idx = argmax(ant_fitnesses)
        best_ant_solution = ant_solutions[best_idx]
        for j in 1:num_features
            if best_ant_solution[j] == 1
                pheromone[j] += Q * (ant_fitnesses[best_idx] / maximum(ant_fitnesses))
                
            end
        end

        if iter % 10 == 0 || iter == 1
            println("Iteration $iter, Best fitness: $best_fitness")
        end
    end

    return best_solution, best_fitness
end


num_features = GENE_SIZE

K = 5   # Number of interactions per feature (higher = harder)

nk = NKLandscape(num_features, K)
best_subset, best_fitness = binary_aco(fitness; num_features=num_features, iterations=100, nk)

println("\nFinal Best subset fitness: ", best_fitness)
println("Best subset: ", best_subset)
