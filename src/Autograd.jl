module Autograd

include("graphs/Graphs.jl")
using .Graphs
export Graph, dfs, topological_sort

include("calculus/Calculus.jl")
using .Calculus
export Scalar, Tensor, backward

include("nn/NN.jl")
using .NN
export Module, Sequential, Linear, ReLU, parameters

end
