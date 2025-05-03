include("../functional.jl")

struct ReLU <: Module
end

function (m::ReLU)(x::Tensor)
    return relu(x)
end

parameters(::ReLU) = []

Base.show(io::IO, m::ReLU) = print(io, "ReLU()")
