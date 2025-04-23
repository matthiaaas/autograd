module NN

using ..Calculus

# struct Parameter

# end

abstract type Module end

include("modules/sequential.jl")
include("modules/linear.jl")
include("modules/relu.jl")

export Module, Sequential, Linear, parameters

end
