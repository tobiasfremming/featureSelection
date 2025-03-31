using Pkg
Pkg.add("PyCall")

module LookupTableModule

using PyCall

# Import Python's pickle module
pickle = pyimport("pickle")

mutable struct LookupTable
    table::Dict{String, Float64}
end

function Base.getindex(self::LookupTable, ind) return self.table[ind] end
function Base.setindex!(self::LookupTable, ele, ind) setindex!(self.table, ele, ind) end
function Base.keys(self::LookupTable) return keys(self.table) end
function Base.values(self::LookupTable) return values(self.table) end
function Base.haskey(self::LookupTable, key) return haskey(self.table, key) end
function Base.delete!(self::LookupTable, key) delete!(self.table, key) end

"""
    create()

Create a new empty lookup table.
"""
function create()
    return LookupTable(Dict{String, Float64}())
end

"""
    get_or_evaluate!(lut::LookupTable, chromosome::Vector{Bool}, fitness_fn::Function)

Returns fitness from the lookup table if it exists; otherwise computes it using `fitness_fn`, stores it, and returns it.
"""
function get_or_evaluate!(lookup_table::LookupTable, chromosome::Vector{Bool}, fitness_fn::Function)
    key = join(map(x -> x ? "1" : "0", chromosome))
    if haskey(lookup_table.table, key)
        return lookup_table.table[key]
    else
        fitness = fitness_fn(chromosome)
        lookup_table.table[key] = fitness
        return fitness
    end
end

"""
    save(lut::LookupTable, path::String)

Save the lookup table to disk using Python's pickle.
"""
function save(lookup_table::LookupTable, path::String)
    # Convert Julia Dict to Python dict
    py_dict = PyDict(lookup_table.table)
    open(path, "w") do f 
        pickle.dump(py_dict, f)
    end
end

"""
    load(path::String) -> LookupTable

Load a lookup table from disk using Python's pickle.
"""
function load(path::String)
    open(path, "r") do f
        py_dict = pickle.load(f)
        # Convert back to Julia Dict
        julia_dict = Dict{String, Float64}()
        for (k, v) in py_dict
            julia_dict[string(k)] = float(v)
        end
        return LookupTable(julia_dict)
    end
end

end # module
