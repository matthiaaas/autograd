mutable struct Tensor{T<:Array}
    value::T
    grad::T
    backward::Function
    dependencies::Set{Tensor}
end

Tensor(value::T, grad::T, backward::Function, dependencies::Set{Tensor}) where {T<:Array} = Tensor{T}(value, grad, backward, dependencies)

Base.show(io::IO, t::Tensor) = print(io, "Tensor($(t.value), grad=$(t.grad))")
