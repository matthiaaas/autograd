using ..Graphs

mutable struct Scalar{T<:Real}
    value::T
    grad::T
    backward::Function
    children::Set{Scalar}
end

Scalar(value::T, children::Set{Scalar}=Set{Scalar}()) where {T<:Real} = Scalar{T}(value, zero(T), () -> nothing, children)

Base.show(io::IO, s::Scalar) = print(io, "Scalar($(s.value), grad=$(s.grad))")

function Base.:+(a::Scalar, b::Scalar)
    y = Scalar(a.value + b.value, Set{Scalar}([a, b]))

    y.backward = () -> begin
        a.grad += y.grad
        b.grad += y.grad
    end

    return y
end

Base.:+(a::Scalar, b::Real) = a + Scalar(b)
Base.:+(a::Real, b::Scalar) = Scalar(a) + b

function Base.:*(a::Scalar, b::Scalar)
    y = Scalar(a.value * b.value, Set{Scalar}([a, b]))

    y.backward = () -> begin
        a.grad += b.value * y.grad
        b.grad += a.value * y.grad
    end

    return y
end

Base.:*(a::Scalar, b::Real) = a * Scalar(b)
Base.:*(a::Real, b::Scalar) = Scalar(a) * b

function Base.:^(a::Scalar, b::Real)
    y = Scalar(a.value^b, Set{Scalar}([a]))

    y.backward = () -> begin
        a.grad += b * a.value^(b - 1) * y.grad
    end

    return y
end

Base.:-(a::Scalar) = -1.0 * a
Base.:-(a::Scalar, b::Scalar) = a + (-b)
Base.:-(a::Scalar, b::Real) = a - Scalar(b)
Base.:-(a::Real, b::Scalar) = Scalar(a) - b

Base.:/(a::Scalar, b::Scalar) = a * (b^(-1))
Base.:/(a::Scalar, b::Real) = a / Scalar(b)
Base.:/(a::Real, b::Scalar) = Scalar(a) / b

Base.:inv(a::Scalar) = Base.:^(a, -1)

Base.length(s::Scalar) = 1

function backward(s::Scalar)
    g = Graph(s, (node::Scalar) -> node.children)

    s.grad = 1.0

    for s in topological_sort(g)
        s.backward()
    end
end
