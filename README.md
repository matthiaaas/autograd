# `autograd`

Autograd engine for scalar or n-dimensional tensor values using either chain rule backpropagation or forward mode autodifferentiation.

> [!WARNING]
> Built for educational purposes.

## Dual Numbers & Forward Mode Automatic Differentiation

```jl
z = Dual(2.0, 3.0) + Dual(4.0, 5.0) # Dual(6.0, 8.0)

f(x) = x^2 + 3x + 2
autodiff(f, 2.0) # 7.0
```

As seen on [Computerphile](https://www.youtube.com/watch?v=QwFLA5TrviI).

## Chain Rule Backpropagation / Reverse Mode Automatic Differentiation

```jl
w = Scalar(2.0)
x = Scalar(3.0)
b = Scalar(1.0)

L = w * x + b # Scalar(7.0, grad=0.0)

backward(L) # ~> w = Scalar(2.0, grad=3.0), b = Scalar(1.0, grad=1.0)
```

```jl
w = Tensor([1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
b = Tensor([1.0; 1.0; 1.0])

x = [1.0; 2.0; 3.0]

y_hat = w * x + b

backward(y_hat)

relu(y_hat)
mse(y_hat)
# etc.
```

As seen on [PyTorch](https://pytorch.org).

## Framework

```jl
model = Sequential(
    Linear(256, 128),
    ReLU(),
    Linear(128, 64),
    Linear(64, 10),
    Sigmoid()
)

criterion = CrossEntropy()
optimizer = Adam(model)

zero_grad(optimizer)
output = model(x)
loss = criterion(output, target)
backward(loss)
step(optimizer)
```
