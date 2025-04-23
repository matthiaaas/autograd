struct Sequential <: Module
    layers::Vector{Module}
end

function Sequential(layers::Module...)
    return Sequential(collect(layers))
end

function (m::Sequential)(x)
    for layer in m.layers
        x = layer(x)
    end

    return x
end

parameters(m::Sequential) = vcat(parameters.(m.layers)...)

Base.show(io::IO, m::Sequential) = print(io, "Sequential($(join(string.(m.layers), ", ")))")
