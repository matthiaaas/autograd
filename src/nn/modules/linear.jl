mutable struct Linear <: Module
    in_features::Int
    out_features::Int

    weight::Tensor{Matrix{Float64}}
    bias::Union{Tensor{Vector{Float64}},Nothing}
end

function Linear(in_features::Int, out_features::Int, bias::Bool=true)
    weight = Tensor(randn(out_features, in_features))
    bias = bias ? Tensor(randn(out_features)) : nothing

    return Linear(in_features, out_features, weight, bias)
end

(m::Linear)(x::Tensor) = m.weight * x .+ (m.bias !== nothing ? m.bias : 0.0)

parameters(m::Linear) = [m.weight; m.bias]

Base.show(io::IO, m::Linear) = print(io, "Linear($(m.in_features), $(m.out_features))")
