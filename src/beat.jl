using Random
using Dates
using FileIO
using DataFrames
using CSV
using Statistics

# === Include your modules ===
include("agent.jl")
include("GA/crossover.jl")
include("GA/mutate.jl")
include("lookup_table.jl")

using .Agent
using .Crossover
using .Mutations
using .LookupTableModule

# === CONFIG ===
DATASET = ARGS[1]
ERROR_TABLE = LookupTableModule.load("../luts/$(DATASET)_err.pickle")
ZERO_KEY = string(join([0 for _ in 1:length(collect(keys(ERROR_TABLE))[1])]))
ZERO_VALUE = maximum(values(ERROR_TABLE)) + 1
SAVE_PATH = "out/$(Dates.format(Dates.now(), "dd.mm.HH.MM"))"

POP_SIZE = 200
GENERATIONS = 100
ELITE_FRACTION = 0.1

# NEAT distance weights
C1 = 1.0
C2 = 1.0
C3 = 0.4
DELTA_T = 3.0

# === Folder setup ===
function create_run_folders(savepath::String)
    mkpath("$(savepath)/plots")
    mkpath("$(savepath)/runs")
end
create_run_folders(SAVE_PATH)

# === Fitness Evaluation ===
function evaluate_fitness!(population::Vector{Chromosome}, error_table::LookupTableModule.LookupTable, zero_key::String, zero_value::Float64)
    for chromo in population
        bitstring = string(join([g.weight > g.bias ? 1 : 0 for g in chromo.genes]))
        if bitstring == zero_key
            chromo.fitness = -zero_value
        else
            chromo.fitness = -get(error_table, bitstring, zero_value)
        end
    end
end

# === Parent Selection ===
function select_parent(population::Vector{Chromosome})
    total_fitness = sum(c.adjusted_fitness for c in population)
    pick = rand() * total_fitness
    current = 0.0
    for c in population
        current += c.adjusted_fitness
        if current >= pick
            return c
        end
    end
    return population[end]  # fallback
end

# === Logger ===
function log_statistics!(population::Vector{Chromosome}, gen::Int, savepath::String)
    fitnesses = [c.fitness for c in population]
    println("Generation $gen | Mean = $(mean(fitnesses)) | Max = $(maximum(fitnesses)) | Min = $(minimum(fitnesses)) | Std = $(std(fitnesses))")

    df = DataFrame(
        generation = gen,
        mean_fitness = mean(fitnesses),
        std_fitness = std(fitnesses),
        max_fitness = maximum(fitnesses),
        min_fitness = minimum(fitnesses),
    )
    CSV.write("$(savepath)/runs/gen_$gen.csv", df)
end

# === Main GA Runner ===
function run_neat_ga!(
    population::Vector{Chromosome},
    generations::Int,
    c1::Float64,
    c2::Float64,
    c3::Float64,
    delta_t::Float64,
    elite_fraction::Float64,
    error_table::LookupTableModule.LookupTable,
    zero_key::String,
    zero_value::Float64,
    savepath::String
)
    for gen in 1:generations
        evaluate_fitness!(population, error_table, zero_key, zero_value)
        species = Agent.speciate_and_fitness_sharing!(population, c1, c2, c3, delta_t)

        sort!(population, by = x -> -x.adjusted_fitness)
        elite_count = Int(round(length(population) * elite_fraction))
        new_population = deepcopy(population[1:elite_count])

        while length(new_population) < length(population)
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = Crossover.crossover(parent1, parent2)
            mutate!(child, max_node_id=100)  # adjust as needed
            push!(new_population, child)
        end

        empty!(population)
        append!(population, new_population)

        log_statistics!(population, gen, savepath)
    end
end

# === Run one trial ===
population = [Agent.create_random_chromosome(10, 0, 10) for _ in 1:POP_SIZE]

run_neat_ga!(
    population,
    GENERATIONS,
    C1, C2, C3,
    DELTA_T,
    ELITE_FRACTION,
    ERROR_TABLE,
    ZERO_KEY,
    ZERO_VALUE,
    SAVE_PATH
)

# === Optional: Description and metadata ===
println("Enter a description of this run: ")
description = readline()
open("$(SAVE_PATH)/index.txt", "w") do f
    write(f, "Dataset: $(DATASET)\n")
    write(f, "Description: $(description)\n")
    write(f, "Population Size: $(POP_SIZE)\n")
    write(f, "Generations: $(GENERATIONS)\n")
    write(f, "Mutation: NEAT-style\n")
    write(f, "Crossover: Historical NEAT-style\n")
end
