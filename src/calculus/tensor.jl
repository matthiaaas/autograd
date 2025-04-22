using ..Graphs

mutable struct Tensor{T}
    value::T
    grad::T
    backward::Function
    children::Set{Tensor}
end

function Tensor(value::T, children::Set{Tensor}=Set{Tensor}()) where {T}
    if isa(value, Number)
        return Tensor{T}(value, zero(T), () -> nothing, children)
    else
        return Tensor{T}(value, zeros(size(value)), () -> nothing, children)
    end
end

Base.show(io::IO, t::Tensor) = print(io, "Tensor($(t.value), grad=$(t.grad))")

function Base.:+(a::Tensor, b::Tensor)
    y = Tensor(a.value + b.value, Set{Tensor}([a, b]))

    y.backward = () -> begin
        a.grad += y.grad
        b.grad += y.grad
    end

    return y
end

Base.:+(a::Tensor, b::Array) = a + Tensor(b)
Base.:+(a::Array, b::Tensor) = Tensor(a) + b
Base.:+(a::Tensor, b::Real) = a + Tensor(b)
Base.:+(a::Real, b::Tensor) = Tensor(a) + b

function Base.:*(a::Tensor, b::Tensor)
    y = Tensor(a.value * b.value, Set{Tensor}([a, b]))

    y.backward = () -> begin
        if ndims(a.value) == 0
            a.grad += sum(y.grad .* b.value)
        else
            a.grad += y.grad * transpose(b.value)
        end

        if ndims(b.value) == 0
            b.grad += sum(transpose(a.value) .* y.grad)
        else
            b.grad += transpose(a.value) * y.grad
        end
    end

    return y
end

Base.:*(a::Tensor, b::Array) = a * Tensor(b)
Base.:*(a::Array, b::Tensor) = Tensor(a) * b
Base.:*(a::Tensor, b::Real) = a * Tensor(b)
Base.:*(a::Real, b::Tensor) = Tensor(a) * b

Base.:-(a::Tensor) = -1.0 * a
Base.:-(a::Tensor, b::Tensor) = a + (-b)
Base.:-(a::Tensor, b::Array) = a - Tensor(b)
Base.:-(a::Array, b::Tensor) = Tensor(a) - b

Base.broadcastable(t::Tensor) = Ref(t)

Base.size(t::Tensor) = size(t.value)
Base.length(t::Tensor) = length(t.value)

function Base.abs(t::Tensor)
    y = Tensor(abs.(t.value), Set{Tensor}([t]))

    y.backward = () -> begin
        t.grad += sign.(t.value) .* y.grad
    end

    return y
end

function Base.sum(t::Tensor)
    y = Tensor(sum(t.value), Set{Tensor}([t]))

    y.backward = () -> begin
        t.grad += ones(size(t.value))
    end

    return y
end

function mean(t::Tensor)
    y = Tensor(mean(t.value), Set{Tensor}([t]))

    y.backward = () -> begin
        t.grad += ones(size(t)) / length(t)
    end

    return y
end

function backward(t::Tensor)
    g = Graph(t, (node::Tensor) -> node.children)

    if isa(t.grad, Number)
        t.grad = one(t.value)
    else
        t.grad = ones(size(t.value))
    end

    for t in topological_sort(g)
        t.backward()
    end
end
