mutable struct Linear{W,B} <: Module
    in_features::Int
    out_features::Int

    weight::Tensor{W}
    bias::Union{Tensor{B},Nothing}
end

function Linear(in_features::Int, out_features::Int, bias::Bool=true)
    if bias
        return Linear{Float64,Float64}(in_features, out_features,
            Tensor(randn(out_features, in_features)),
            Tensor(randn(out_features)))
    else
        return Linear{Float64,Nothing}(in_features, out_features,
            Tensor(randn(out_features, in_features)), nothing)
    end
end

Base.show(io::IO, m::Linear) = print(io, "Linear($m.in_features, $m.out_features)")
