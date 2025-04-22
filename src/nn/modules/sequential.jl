struct Sequential{T<:Module} <: Module
    modules::Vector{T}
end

function Sequential(modules::T...) where {T<:Module}
    return Sequential{T}(modules)
end
