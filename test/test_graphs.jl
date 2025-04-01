using Test

include("../graphs.jl")

using .Graphs

@testset "Graphs Tests" begin
    @testset "Node initialization" begin
        n1 = Node(1)
        n2 = Node(2, Set{Node}([n1]))
        n3 = Node(3, Set{Node}([n1, n2]))

        @test n1.value == 1
        @test n2.value == 2
        @test n3.value == 3

        @test length(n2.children) == 1
        @test length(n3.children) == 2
    end

    # @testset "Graph structure" begin
    #     g = Graph(n3, (node::Node) -> node.children)

    #     @test g.root.value == 3
    # end
end
