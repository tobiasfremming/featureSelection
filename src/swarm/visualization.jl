module PSOPlots



using Plots
using ColorSchemes
using Distances
using StatsBase               # countmap / histogram helpers
using MultivariateStats        # PCA for the flight‑path animation
using StatsPlots  
# Re‑export the most popular routines so users can do
#   using .PSOPlots; convergence_plot(...)
export convergence_plot, heatmap_plot, freq_plot,
       plot_diversity, fitness_dist_plot, plot_mutation,
       flip_heatmap, velocity_histogram, parallel_coords,
       flight_animation

"""
    convergence_plot(best_hist; true_value = nothing)

Line‑chart of the best‑so‑far fitness the swarm has achieved.
Pass the true optimum via keyword `true_value` to draw a dashed
reference line.
"""
function convergence_plot(best_hist; true_value = nothing)
    iters = 1:length(best_hist)
    plt   = plot(iters, best_hist;
                 xlabel = "iteration",
                 ylabel = "best fitness",
                 title  = "Convergence curve",
                 lw     = 2,
                 legend = :topright)
    if !isnothing(true_value)
        hline!(plt, [true_value]; ls = :dash, label = "true optimum")
    end
    display(plt); return plt
end

"""
    heatmap_plot(feat_hist)

`feat_hist` must be a *Bool* matrix with shape *(iters, features)* that
contains the **global best bit‑string at every iteration**.  White ⇒ bit
OFF, Green ⇒ bit ON.
"""
function heatmap_plot(feat_hist::AbstractMatrix{Bool})
    heatmap( feat_hist',                        # features on Y‑axis
             c      = cgrad(:Dark2_8, 2, categorical = true),
             xlabel = "iteration",
             ylabel = "feature index",
             yflip  = true,
             legend = false,
             title  = "Feature ON (🟩) / OFF (⬜) through time") |> display
end

"""
    freq_plot(feat_hist; top = 20)

Bar‑chart of the *top* «`top`» most frequently selected features in the
*global best* history.
"""
function freq_plot(feat_hist::AbstractMatrix{Bool}; top::Int = 20)
    counts = vec(sum(feat_hist; dims = 1))                # hits per bit
    order  = sortperm(counts; rev = true)
    top    = min(top, length(order))
    idxs   = order[1:top]
    bar( 1:top, counts[idxs];
         xlabel = "feature (ranked)", ylabel = "hits",
         xticks = (1:top, string.(idxs)),
         legend = false,
         title  = "Most frequently selected features") |> display
end

"""
    diversity_plot(diversity_vec)

`diversity_vec` is the mean pair‑wise Hamming distance of the swarm each
iteration.
"""
function plot_diversity(diversity_vec)
    plot(diversity_vec;
         xlabel = "iteration", ylabel = "mean Hamming distance",
         title  = "Swarm diversity", lw = 2) |> display
end

"""
    fitness_dist_plot(fit_hist)

`fit_hist` = Matrix(iters × particles) with every particle's fitness.
Shows a violin/box plot per iteration.
"""
function fitness_dist_plot(fit_hist::AbstractMatrix)
                         # lazy‑load to avoid hard dep
    iters = size(fit_hist, 1)
    violin( repeat(1:iters, inner = size(fit_hist,2)), vec(fit_hist);
             xlabel = "iteration", ylabel = "fitness",
             title  = "Fitness distribution", legend = false) |> display
end

"""
    mutation_plot(A_hist)

Plot the A‑mutation probability schedule that INBPSO used.
"""
function plot_mutation(A_hist)
    plot(A_hist; xlabel = "iteration", ylabel = "A (mutation prob.)",
         title = "INBPSO mutation schedule", lw = 2) |> display
end

"""
    flip_heatmap(flip_hist)

`flip_hist` – Bool matrix *(iters, features)* where `true` marks a *bit
flip* in the global best between successive iterations.
"""
function flip_heatmap(flip_hist::AbstractMatrix{Bool})
    heatmap( flip_hist'; c = cgrad(:roma, 2, categorical = true),
             xlabel = "iteration", ylabel = "feature index", yflip = true,
             legend = false, title = "Bit flips in global best") |> display
end

"""
    velocity_histogram(v_hist; nbins = 30)

`v_hist` = vector of all velocity magnitudes collected over the run.
"""
function velocity_histogram(v_hist; nbins = 30)
    histogram(abs.(v_hist); nbins, xlabel = "|v|", ylabel = "count",
              title = "Velocity magnitude distribution") |> display
end

"""
    parallel_coords(best_bits)

Stick plot of the final best bit‑string.
"""
function parallel_coords(bits::AbstractVector{Bool})
    plot(1:length(bits), Int.(bits); seriestype = :sticks, ms = 3,
         xlabel = "feature", ylabel = "on / off",
         yticks = ([0,1],["0","1"]), title = "Final solution") |> display
end



"""
    flight_animation(bitlog; fps = 15, file = "swarm_flight.gif")

*bitlog* must be a **Bool array** with size *(iters, particles, dim)*
that stores every particle's bit‑string each iteration (see README in
code comments).  Creates a `.gif` and returns the `Animation` object so
you can also `gif(anim, io)` yourself.
"""
function flight_animation(bitlog::BitMatrix; fps::Int = 12,
        file::String = "flight.gif")
    X = Float64.(bitlog)             # (iters × features)
    stds = vec(std(X; dims = 1))
    keep = findall(>(0), stds)       # columns with any variance
    X    = X[:, keep]

    # Safe guard: if < 2 dims left just duplicate the single column
    if size(X, 2) == 0
    X = hcat(zeros(size(X,1)), zeros(size(X,1)))
    elseif size(X, 2) == 1
    X = hcat(X, X)               # duplicate → flat line
    end

    M   = fit(PCA, X; method = :svd, maxoutdim = 2)
    Y   = transform(M, X)            # (iters × 2) coordinates

    frames = @animate for i in 1:size(Y,1)
    scatter(Y[1:i, 1], Y[1:i, 2];
    marker  = :circle,
    ms      = 3,
    line    = (:path, 1, :blue),
    xlabel  = "PC1",
    ylabel  = "PC2",
    title   = "global-best flight-path",
    leg     = false,
    xlims   = extrema(Y[:,1]) .+ (-0.1,0.1),
    ylims   = extrema(Y[:,2]) .+ (-0.1,0.1))
    end every 2   # show every 2nd point to keep the GIF small

    gif(frames, file; fps = fps)
    @info "saved flight animation → $file"
    return nothing
end

end # module
