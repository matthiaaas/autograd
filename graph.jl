module Graph

export Node, dfs

struct Node{T}
    value::T
    children::Set{Node}
end

Node(value::T, children::Set{Node}=Set{Node}()) where {T} = Node{T}(value, children)

Base.show(io::IO, n::Node) = print(io, "Node($(n.value))")

function dfs(node::Node)
    visited = Set{Node}()
    stack = []

    function visit(n::Node)
        if n in visited
            return
        end

        push!(visited, n)

        for child in n.children
            visit(child)
        end

        push!(stack, n)
    end

    visit(node)

    return stack
end

w = Node("w")
x = Node("x")
g = Node("g", Set{Node}([w, x]))
b = Node("b")
L = Node("L", Set{Node}([b, g]))

println("Visited nodes: ", dfs(L))

end
