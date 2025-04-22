module NN

using ..Calculus

# struct Parameter

# end

abstract type Module end

include("modules/sequential.jl")
include("modules/linear.jl")


function mse(pred, target)
    return (pred - target)^2
end

export Module, Sequential, Linear

end
