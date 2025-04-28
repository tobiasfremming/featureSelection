

using Random, Statistics, Printf, ThreadsX, Dates
# ──────────────────────────────────────────────────────────────────
#  LOOK-UP TABLE (heart / cancer / diabetes)  ──────────────────────
# ──────────────────────────────────────────────────────────────────
using CSV, DataFrames                        # add these two

const DATASET = get(ENV, "DATASET", "cancer")   # choose set via ENV var

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
@inline function fitness(bits::BitVector)::Float64
    return LUT_FITNESS[bits]      # negate because PSO maximises
end
# ──────────────────────────────────────────────────────────────────


"""
    best_index(swarm) -> (idx, best_fitness)

Return the index of the particle with the highest `fP`
and that fitness value (works on all Julia versions).
"""
function best_index(swarm)
    best_idx = 1
    best_fit = swarm[1].fP
    @inbounds for i in 2:length(swarm)
        fi = swarm[i].fP
        if fi < best_fit
            best_fit = fi
            best_idx = i
        end
    end
    return best_idx, best_fit
end


@inline sigmoid(x) = 1 / (1 + exp(-x))                   # classic
@inline Sprime(v)  = abs(tanh(v / 2))                    # NBPSO map ∈ [0,1]

# INBPSO A‑mutation schedule (eq. 11)
@inline function A_mutation(F::Int, T::Real, k::Real=1)
    return 1 - exp(-k * F / T)        # ∈ [0,1)
end



"""
    fitness(bitvec::BitVector) -> Float64

Toy fitness: maximise number of 1‑bits ("Max‑Ones").
Replace this with your own evaluator.
"""


function fitness3(bits::BitVector)
   
    ones = 0
    zeros = 0
    for i in 1:length(bits)
        if bits[i] == 1
            ones += 0.7
        else
            zeros += 1
        end
    end
    return max(ones, zeros) / length(bits)
end


mutable struct Particle
    X   :: BitVector      # current position (bitstring)
    V   :: Vector{Float64}# velocity (real)
    P   :: BitVector      # personal best position
    fP  :: Float64        # fitness of P
end

# neighbourhood topology 

abstract type AbstractTopology end

struct GlobalTopology <: AbstractTopology  # everyone sees everyone
    n::Int
end
get_best(::GlobalTopology, swarm, _) = best_index(swarm)[1]

struct RingTopology <: AbstractTopology
    neighbours::Vector{NTuple{2,Int}}    # 2‑ring (left/right)
end
function RingTopology(m::Int)
    neigh = [(mod1(i-1,m), mod1(i+1,m)) for i in 1:m]
    return RingTopology(neigh)
end
function get_best(topo::RingTopology, swarm, i)
    idxs = (i, topo.neighbours[i]...)
    best = idxs[argmax(swarm[j].fP for j in idxs)]
    return best
end

# Swarm initialisation                                                    #


function init_swarm(rng::AbstractRNG, m::Int, d::Int, v_max::Float64)
    swarm = Vector{Particle}(undef, m)
    @inbounds for i in 1:m
        X  = BitVector(rand(rng, Bool, d))
        V  = rand(rng, d) .* 2v_max .- v_max     # uniform (−v_max,v_max)
        f  = fitness(X)
        swarm[i] = Particle(X, V, copy(X), f)
    end
    return swarm
end


# 5.  Binary PSO (supports NBPSO + INBPSO)                                    

"""
    binary_pso(fitness; features, particles=50, iters=200, rng,              
               v_max   = 6.0,  w_start=0.9, w_end=0.4, c1=2.2, c2=2.5,       
               topology=:global, stagn_limit=20, swarm_stagn_limit=40,       
               T_factor = 5.0) → NamedTuple

`topology`  : :global | :ring
Returns     : (best, best_fitness, history)
"""
function binary_pso(;                                  
        features::Int, particles::Int = 50, iters::Int = 200,
        rng::AbstractRNG = Random.default_rng(),
        v_max::Float64   = 2.0, w_start=0.9, w_end=0.4, c1=2.2, c2=2.5,
        topology::Symbol = :ring, # :global | :ring
        stagn_limit::Int = 15, swarm_stagn_limit::Int = 40,
        T_factor::Real   = 5.0)

    # prepare objects 
    swarm = init_swarm(rng, particles, features, v_max)

    topo = topology === :global ? GlobalTopology(particles) : RingTopology(particles)

    best_hist = Float64[]     # record best each iter
    

    # swarm‑level stagnation counter
    last_best  = maximum(p.fP for p in swarm)
    F_swarm    = 0
    T          = T_factor * features

    # per‑particle stagnation counters
    stagn_cnt  = zeros(Int, particles)

    w_schedule = range(w_start, w_end; length=iters)

    # -------------------------------------------------------------------------
    for iter in 1:iters
        w = w_schedule[iter]

        Threads.@threads for i in 1:particles
            p = swarm[i]

            # --- choose neighbourhood best -----------------------------------
            best_idx = get_best(topo, swarm, i)
            G = swarm[best_idx].P

            # --- velocity + position update ----------------------------------
            @inbounds for j in 1:features
                r1, r2 = rand(rng), rand(rng)
                vij = w*p.V[j] +               # inertia
                      c1*r1*(p.P[j]-p.X[j]) +  # cognitive
                      c2*r2*(G[j]  -p.X[j])    # social
                vij = clamp(vij, -v_max, v_max)
                p.V[j] = vij

                if rand(rng) < Sprime(vij)     # NBPSO flip rule
                    p.X[j] ⊻= true
                end
                # if rand(rng) < sigmoid(vij)                
                #     p.X[j] ⊻= true
                # end
            end
            # fitness / personal best 
            f_curr = fitness(p.X)
            if f_curr < p.fP
                p.P  = copy(p.X)
                p.fP = f_curr
                stagn_cnt[i] = 0
            else
                stagn_cnt[i] += 1
            end

            # kick stagnant particles
            if stagn_cnt[i] ≥ stagn_limit
                p.V .= rand(rng, features) .* 2v_max .- v_max
                idx = rand(rng, 1:features)
                p.X[idx] ⊻= true
                stagn_cnt[i] = 0
            end
        end  # threads loop

        # get global best
        (gbest_idx, gbest_f) = best_index(swarm)
        push!(best_hist, gbest_f)

        # swarm stagnation & INBPSO mutation 
        if gbest_f == last_best
            F_swarm += 1
        else
            F_swarm  = 0
            last_best = gbest_f
        end

        A = A_mutation(F_swarm, T)
        
        if A > 0
            @inbounds for p in swarm
                for j in 1:features
                    if rand(rng) < A / features
                        p.X[j] ⊻= true
                    end
                end
            end
        end

        # print progress 
        @printf("iter %4d  |  best = %.6f  |  A = %.3f\n", iter, gbest_f, A)
    end  # main loop

    #gbest_f, gbest_idx = findmax(swarm, by = s -> s.fP)
    (gbest_idx, gbest_f) = best_index(swarm)
    gbest = copy(swarm[gbest_idx].P)

    return (best = gbest, fitness = gbest_f, history = best_hist)
end




println("[", Dates.format(now(), "HH:MM:ss"), "]  demo run…\n")
result = binary_pso(features = GENE_SIZE, particles = 60, iters = 400,
                    topology = :ring,
                    rng = MersenneTwister(2025))

println("\nBest fitness = ", result.fitness)
println("Best solution = ", result.best)
