using Test

include("../calculus.jl")

using .AutogradCalculus

@testset "Autograd Calculus Tests" begin
    @testset "Scalar values arithmetic" begin
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = Scalar(1.0)

        @test (a + b).value == 5.0
        @test (a - b).value == -1.0
        @test (a * b).value == 6.0
        @test (a / c).value == 2.0

        @test (4 * a + 2.0 * b + 1).value == 15.0
        @test (a * b + c).value == 7.0
        @test (a^3).value == 8.0
        @test ((a + b)^2).value == 25.0
        @test (a * b^4 + 6 * c).value == 168.0
    end

    # @testset "Backpropagation" begin
    #     a = Scalar(2.0)
    #     b = Scalar(3.0)
    #     c = Scalar(1.0)
    #     d = Scalar(4.0)
    #     e = Scalar(5.0)

    #     L1 = a * b + c * d + e
    #     backward(L1)

    #     L2 = a^2 + b - c
    #     backward(L2)

    #     @test a.grad == 3.0
    #     @test b.grad == 2.0
    # end
end
