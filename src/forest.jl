using DecisionTree
using Random
using Statistics
using CSV
using DataFrames

function RF(train_labels, train_features, test_labels, test_features)
    model = build_forest(train_labels, train_features, -1, 30, 0.7, -1, 2, 2; rng=MersenneTwister(456))
    preds = apply_forest(model, test_features)
    acc = mean(preds .== test_labels)
    return acc
end

# function runModel(features, labels, bitmask_integer::Int)
#     bitmask = [b == '1' for b in last(bitstring(bitmask_integer), size(features, 2))]
#     n = size(features, 1)
    
#     accuracies = []

#     for _ in 1:30
#         train_indices = randperm(MersenneTwister(123), n)[1:Int64(floor(0.7*n))]
#         test_indices = setdiff(1:n, train_indices)

#         acc = RF(labels[train_indices], features[train_indices, bitmask], labels[test_indices], features[test_indices, bitmask])
#         push!(accuracies, acc)
#     end

#     return mean(accuracies)
# end

function runModel(features, labels, bitmask)
    n = size(features, 1)
    
    accuracies = []

    rng = MersenneTwister(123)
    for _ in 1:30
        train_indices = randperm(rng, n)[1:Int64(floor(0.7*n))]
        test_indices = setdiff(1:n, train_indices)

        acc = RF(labels[train_indices], features[train_indices, findall(bitmask)], labels[test_indices], features[test_indices, findall(bitmask)])
        push!(accuracies, acc)
    end

    return mean(accuracies)
end

function loadClevelandData()
    df = CSV.read("../data_sets/processed.cleveland.data", DataFrame, header=false)
    return Matrix{Float64}(df[:, 1:end-1]), df[:, end]
end

function loadZooData()
    df = CSV.read("../data_sets/zoo.data", DataFrame, header=false)
    return Matrix{Float64}(df[:, 2:end-1]), df[:, end]
end

function loadLetterData()
    df = CSV.read("../data_sets/letter-recognition.data", DataFrame, header=false)
    return Matrix{Float64}(df[:, 2:end]), df[:, 1]
end

CLEVELAND_DATASET = loadClevelandData()
ZOO_DATASET = loadZooData()
LETTER_DATASET = loadLetterData()

# function computeClevelandFitness(bitmask_integer::Int)
#     return 1 - runModel(CLEVELAND_DATASET..., bitmask_integer)
# end 

# function computeZooFitness(bitmask_integer::Int)
#     bitmask = [b == '1' for b in bitstring(bitmask_integer)]
#     return 1 - runModel(ZOO_DATASET..., bitmask_integer) + count(bitmask)/64
# end

# function computeLetterFitness(bitmask_integer::Int)
#     bitmask = [b == '1' for b in bitstring(bitmask_integer)]
#     return 1 - runModel(LETTER_DATASET..., bitmask_integer) + count(bitmask)/8
# end

function computeClevelandFitness(bitmask)
    return 1 - runModel(CLEVELAND_DATASET..., bitmask)
end 

function computeZooFitness(bitmask)
    return 1 - runModel(ZOO_DATASET..., bitmask) + count(bitmask)/64
end

function computeLetterFitness(bitmask)
    return 1 - runModel(LETTER_DATASET..., bitmask) + count(bitmask)/8
end