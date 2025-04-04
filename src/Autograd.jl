module Autograd

include("graphs/Graphs.jl")

using .Graphs

include("calculus/Calculus.jl")

using .Calculus

export Scalar, Tensor, Graph, dfs, topological_sort, backward

end
