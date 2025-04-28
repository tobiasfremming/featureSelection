

using Random, Statistics, Printf, ThreadsX, Dates
# ──────────────────────────────────────────────────────────────────
#  LOOK-UP TABLE (heart / cancer / diabetes)  ──────────────────────
# ──────────────────────────────────────────────────────────────────
using CSV, DataFrames                        # add these two
include("visualization.jl")
using .PSOPlots  

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

# Swarm initialisation                                                    


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


# Binary PSO (supports NBPSO + INBPSO)                                    

"""
    binary_pso(fitness; features, particles=50, iters=200, rng,              
               v_max   = 6.0,  w_start=0.9, w_end=0.4, c1=2.2, c2=2.5,       
               topology=:global, stagn_limit=20, swarm_stagn_limit=40,       
               T_factor = 5.0) → NamedTuple

`topology`  : :global | :ring
Returns     : 
    - `best_bits`  : the best bitstring found
    - `best_fit`   : the fitness of that bitstring
    - `best_hist`  : the best fitness curve
    - `bit_hist`   : the global-best bits (1 row per iter)
    - `fit_hist`   : the full pop fitness (1 row per iter)
    - `div_hist`   : diversity (Hamming mean) per iter
    - `A_hist`     : INBPSO A schedule
    - `v_hist`     : mean |v| per iter
"""

function binary_pso(;
        features::Int,
        particles::Int      = 50,
        iters::Int          = 200,
        rng::AbstractRNG    = Random.default_rng(),
        v_max::Float64      = 2.0,
        w_start             = 0.9,
        w_end               = 0.4,
        c1::Float64         = 2.2,
        c2::Float64         = 2.5,
        topology::Symbol    = :ring,  # :global | :ring
        stagn_limit::Int    = 15,
        T_factor::Real      = 5.0)

    # ---------------- initialisation -----------------------------------------
    swarm   = init_swarm(rng, particles, features, v_max)
    topo    = topology === :global ? GlobalTopology(particles) : RingTopology(particles)

    # ~~ histories we are going to log ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    best_hist  =  Vector{Float64}(undef, iters)          # best fitness curve
    bit_hist   =  falses(iters, features)                # global-best bits
    fit_hist   =  Vector{Vector{Float64}}(undef, iters)  # full pop fitness
    div_hist   =  Vector{Float64}(undef, iters)          # diversity
    A_hist     =  Vector{Float64}(undef, iters)          # INBPSO A schedule
    v_hist     =  Vector{Float64}(undef, iters)          # mean |v| per iter

    last_best  = maximum(p.fP for p in swarm)
    F_swarm    = 0
    T          = T_factor * features
    stagn_cnt  = zeros(Int, particles)
    w_schedule = range(w_start, w_end; length = iters)

    # ---------------- helper for visualization--------------------------------------------------
    hamming_mean(pop) = begin
        # mean pair-wise Hamming distance (O(n²) but n=50~60)
        d_sum = 0
        cnt   = 0
        @inbounds for i in 1:length(pop)-1, j in i+1:length(pop)
            # d_sum += count(!=, pop[i].X, pop[j].X)
            d_sum += sum(pop[i].X .!= pop[j].X)
            cnt   += 1
        end
        d_sum / (cnt * features)
    end

    # ---------------- main loop ----------------------------------------------
    for iter in 1:iters
        w = w_schedule[iter]

        Threads.@threads for i in 1:particles
            p       = swarm[i]
            best_nb = get_best(topo, swarm, i)
            G       = swarm[best_nb].P        # neighbourhood best

            # velocity + position update
            @inbounds for j in 1:features
                r1, r2 = rand(rng), rand(rng)
                vij = w*p.V[j] +
                    c1*r1*(p.P[j]-p.X[j]) +
                    c2*r2*(G[j]  -p.X[j])
                vij = clamp(vij, -v_max, v_max)
                p.V[j] = vij

                rand(rng) < Sprime(vij) && (p.X[j] ⊻= true)
            end

            # personal best update
            f_curr = fitness(p.X)
            if f_curr < p.fP
                p.P  = copy(p.X)
                p.fP = f_curr
                stagn_cnt[i] = 0
            else
                stagn_cnt[i] += 1
            end

            # kick stagnant particle
            stagn_cnt[i] ≥ stagn_limit && begin
                p.V .= rand(rng, features) .* 2v_max .- v_max
                p.X[rand(rng, 1:features)] ⊻= true
                stagn_cnt[i] = 0
            end
        end # thread

        # ----- global statistics for this iteration --------------------------
        (gbest_idx, gbest_f) = best_index(swarm)
        gbits                = swarm[gbest_idx].P

        best_hist[iter] = gbest_f
        bit_hist[iter, :] .= gbits
        fit_hist[iter]   = [p.fP for p in swarm]
        div_hist[iter]   = hamming_mean(swarm)
        v_hist[iter]     = mean(abs.(vcat((p.V for p in swarm)...)))

        # INBPSO mutation probability
        if gbest_f == last_best
            F_swarm += 1
        else
            F_swarm  = 0
            last_best = gbest_f
        end
        A = 1 - exp(-F_swarm / T)               # eq. 11 (k=1)
        A_hist[iter] = A

        if A > 0
            @inbounds for p in swarm, j in 1:features
                rand(rng) < A / features && (p.X[j] ⊻= true)
            end
        end

        @printf("iter %4d | best = %.6f | A = %.3f\n", iter, gbest_f, A)
    end # iterations

    (gbest_idx, gbest_f) = best_index(swarm)

    return ( best_bits = copy(swarm[gbest_idx].P),
            best_fit  = gbest_f,
            best_hist = best_hist,
            bit_hist  = bit_hist,
            fit_hist  = fit_hist,
            div_hist  = div_hist,
            A_hist    = A_hist,
            v_hist    = v_hist )
end





println("[", Dates.format(now(), "HH:MM:ss"), "]  demo run…\n")




result = binary_pso(features = GENE_SIZE,
                    particles = 60,
                    iters = 400,
                    topology = :ring,
                    rng = MersenneTwister(2025))
println("Best fitness: ", result.best_fit)
println("Best bitstring: ", result.best_bits)

using .PSOPlots                                   # the module we built
PSOPlots.convergence_plot(result.best_hist;
                 true_value = minimum(values(LUT_FITNESS)))

PSOPlots.heatmap_plot(result.bit_hist).
PSOPlots.freq_plot(result.bit_hist; top = GENE_SIZE)
PSOPlots.plot_diversity(result.div_hist)
PSOPlots.plot_mutation(result.A_hist)
PSOPlots.velocity_histogram(result.v_hist)
mat = reduce(vcat, permutedims.(result.fit_hist)) 
PSOPlots.fitness_dist_plot(mat)                # needs StatsPlots
PSOPlots.parallel_coords(result.best_bits)
PSOPlots.flight_animation(result.bit_hist; fps = 15, file = "swarm_flight.gif")




