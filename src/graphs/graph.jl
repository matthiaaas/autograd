struct Graph{T,C}
    root::T
    get_children::Function
end

Graph(root::T, get_children::Function) where {T} = Graph{T,typeof(get_children(root))}(root, get_children)

function dfs(graph::Graph)
    visited = Set{Any}()
    stack = Vector{Tuple{Any,Bool}}([(graph.root, false)])
    results = Vector{Any}()

    while !isempty(stack)
        node, flagged = pop!(stack)

        if node in visited
            continue
        end

        if flagged
            push!(visited, node)
            push!(results, node)
            continue
        end

        push!(stack, (node, true))

        for child in graph.get_children(node)
            if child ∉ visited
                push!(stack, (child, false))
            end
        end
    end

    return results
end

function topological_sort(graph::Graph)
    return reverse(dfs(graph))
end
