module Graphs

export Graph, Node, dfs, topological_sort

struct Node{T}
    value::T
    children::Set{Node}
end

Node(value::T, children::Set{Node}=Set{Node}()) where {T} = Node{T}(value, children)

Base.show(io::IO, n::Node) = print(io, "Node($(n.value))")

struct Graph{T,C}
    root::T
    get_children::Function
end

Graph(root::T, get_children::Function) where {T} = Graph{T,typeof(get_children(root))}(root, get_children)

function dfs(graph::Graph)
    visited = Set{Node}()
    stack = [graph.root]
    results = []

    while !isempty(stack)
        node = pop!(stack)
        if !(node in visited)
            push!(results, node)
            push!(visited, node)
            for child in graph.get_children(node)
                push!(stack, child)
            end
        end
    end

    return results
end

function topological_sort(graph::Graph)
    return reverse(dfs(graph))
end

end
