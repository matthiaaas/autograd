struct Node{T}
    value::T
    children::Set{Node}
end

Node(value::T, children::Set{Node}=Set{Node}()) where {T} = Node{T}(value, children)

Base.show(io::IO, n::Node) = print(io, "Node($(n.value))")
