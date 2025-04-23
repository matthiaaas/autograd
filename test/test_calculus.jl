using Test

include("../src/Autograd.jl")

using .Autograd.Calculus

@testset "Autograd Calculus Tests" begin
    @testset "Scalar values arithmetic" begin
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = Scalar(1.0)

        @test (a + b) == 5.0
        @test (a - b) == -1.0
        @test (a * b) == 6.0
        @test (a / c) == 2.0

        @test (4 * a + 2.0 * b + 1) == 15.0
        @test (a * b + c) == 7.0
        @test (a^3) == 8.0
        @test ((a + b)^2) == 25.0
        @test (a * b^4 + 6 * c) == 168.0
    end

    @testset "Scalar backpropagation" begin
        a = Scalar(2.0)
        b = Scalar(3.0)
        c = Scalar(1.0)
        d = Scalar(4.0)
        e = Scalar(5.0)

        L = a * b + c * d + e
        backward(L)

        @test L.grad == 1.0
        @test a.grad == b
        @test b.grad == a
        @test c.grad == d
        @test d.grad == c
        @test e.grad == 1.0
    end

    @testset "Basic Tensor matrix arithmetic and calculus" begin
        a = Tensor([
            1.0 2.0 3.0;
            4.0 5.0 6.0
        ])
        b = Tensor([
            2.0 3.0;
            4.0 5.0;
            6.0 7.0
        ])

        c = a * b
        backward(c)

        @test c == Tensor([
            28.0 34.0;
            64.0 79.0
        ])
        @test a.grad == Tensor([
            5.0 9.0 13.0;
            5.0 9.0 13.0
        ])
        @test b.grad == Tensor([
            5.0 5.0;
            7.0 7.0;
            9.0 9.0
        ])
    end

    @testset "Advanced Tensor arithmetic and calculus" begin
        # TODO

        a = Tensor(1.0)
        b = Tensor(4.5)
        c = Tensor(2.0)
        d = Tensor(3.0)
        e = Tensor(-2.0)

        @test (a + b * c - d) == 7.0
    end
end
