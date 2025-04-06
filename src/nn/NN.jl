module NN

using ..Calculus

# struct Parameter

# end

# struct Module

# end

# struct Linear{T} <: Module

# end

function mse(pred, target)
    return (pred - target)
end

export mse

end
