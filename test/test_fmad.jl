using Test

include("../fmad.jl")

using .ForwardModeAutodiff

@testset "FMAD Tests" begin
    @testset "Dual number arithmetic" begin
        a = Dual(2.0, 3.0)
        b = Dual(4.0, 5.0)

        @test a + b == Dual(6.0, 8.0)
        @test a - b == Dual(-2.0, -2.0)
        @test a * b == Dual(8.0, 22.0)
        @test a / b == Dual(0.5, 0.125)
        @test a^2 == Dual(4.0, 12.0)

        @test a + 2b == Dual(10.0, 13.0)
    end

    @testset "Differentiation" begin
        f(x) = x^2 + 3x + 2
        g(x) = sin(x)

        @test autodiff(f, 2.0) == 7.0
        @test autodiff(g, 0.0) == 1.0
    end
end
