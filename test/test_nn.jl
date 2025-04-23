using Test
using Random

include("../src/Autograd.jl")

using .Autograd

@testset "Autograd NN Tests" begin
    Random.seed!(1234)

    @testset "Explicit Scalar linear regression" begin
        # Dataset
        true_w = 3.5
        true_b = 2.5
        f(x) = true_w * x + true_b

        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [f(x) for x in xs]

        # Model
        w = Scalar(1.0)
        b = Scalar(0.0)
        model(x) = w * x + b

        # Training
        lr = 0.05
        loss = Inf

        for epoch in 1:100
            for (x, y) in zip(xs, ys)
                pred = model(x)
                loss = (pred - y)^2
                backward(loss)

                w -= lr * w.grad
                b -= lr * b.grad

                w.grad = 0.0
                b.grad = 0.0
            end
        end

        @test abs(w.value - true_w) < 1e-2
        @test abs(b.value - true_b) < 1e-2
        @test abs(loss).value < 1e-2
    end

    @testset "Explicit Tensor affine linear transformation" begin
        # Dataset
        true_w = Tensor([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])
        true_b = Tensor([0.0; 0.0; 1.0])
        f(x) = true_w * x + true_b

        xs = [
            Tensor([1.0; 2.0; 3.0]),
            Tensor([4.0; 5.0; 6.0]),
            Tensor([7.0; 8.0; 9.0]),
        ]
        ys = [f(x) for x in xs]

        # Model
        w = Tensor([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
        b = Tensor([1.0; 1.0; 1.0])
        model(x) = w * x + b

        # Training
        lr = 0.002
        loss = Inf
        L1(pred, target) = sum(abs(pred - target))

        for epoch in 1:500
            for (x, y) in zip(xs, ys)
                pred = model(x)
                loss = L1(pred, y)
                backward(loss)

                w -= lr * w.grad
                b -= lr * b.grad

                w.grad = zeros(size(w))
                b.grad = zeros(size(b))
            end
        end

        @test all(isapprox.(w.value, true_w.value, atol=0.8))
        @test abs(loss) < 0.2
    end

    @testset "Linear" begin
        model = Sequential(
            Linear(8, 4),
            Linear(4, 2),
        )
        println("Model: $(model)")

        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        ŷ = model(x)
        println("ŷ: $(ŷ)")

        y = Tensor([1.0, 2.0])
        loss = sum(abs(ŷ - y))
        println("Loss: $(loss)")

        backward(loss)

        println("Parameters: $(parameters(model))")
    end
end
