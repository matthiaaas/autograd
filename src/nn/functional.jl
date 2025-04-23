function relu(x::Tensor)
    y = Tensor(max.(x.value, 0.0), Set{Tensor}([x]))

    y.backward = () -> begin
        x.grad += (y.value .> 0) .* y.grad
    end

    return y
end
