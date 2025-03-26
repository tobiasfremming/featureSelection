

module Agent
#  resetting the list every generation as opposed to keeping a growing list of mutations throughout evolution is sufficient to prevent an explosion of innovation numbers. 



mutable struct Expression
    from::Int64
    to::Int64
end

mutable struct Gene
    innovation::Int64
    expression::Expression
    enabled::Bool
end


global innovations = Dict{int, Expression}()

mutable struct Chromosome
    genes::Vector{Gene}
    fitness::Float64
    adjusted_fitness::Float64
end


function create_random_Expression(start::Int64, stop::Int64)
    from = rand(start:stop)
    to = rand(start:stop)
    return Expression(from, to)
end

function create_gene(enabled::Bool, from::Int64, to::Int64)
    expression = create_random_Expression(from, to)
    if (haskey(innovations, expression))
        innovation = innovations[expression]
    else
        innovation = length(innovations) + 1
        innovations[expression] = innovation
    end

    return Gene(innovation, expression, enabled)
end




end





