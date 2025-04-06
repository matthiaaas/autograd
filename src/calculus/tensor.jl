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
        a.grad += y.grad * transpose(b.value)
        b.grad += transpose(a.value) * y.grad
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

Base.size(t::Tensor) = size(t.value)

function backward(t::Tensor)
    g = Graph(t, (node::Tensor) -> node.children)

    t.grad = ones(size(t.value))

    for t in topological_sort(g)
        t.backward()
    end
end
