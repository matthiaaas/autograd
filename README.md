# `autograd`

Autograd for scalar values using either chain rule backpropagation or forward mode autodifferentiation.

## Dual Numbers & Forward Mode Automatic Differentiation

```jl
z = Dual(2.0, 3.0) + Dual(4.0, 5.0) # Dual(6.0, 8.0)

f(x) = x^2 + 3x + 2
autodiff(f, 2.0) # 7.0
```

As seen on [Computerphile](https://www.youtube.com/watch?v=QwFLA5TrviI).

## Chain Rule Backpropagation

```jl
w = Scalar(2.0)
x = Scalar(3.0)
b = Scalar(1.0)

L = w * x + b # Scalar(7.0, grad=0.0)

backward(L) # ~> w = Scalar(2.0, grad=3.0), b = Scalar(1.0, grad=1.0)
```

### Forward pass

$$
L = w * x + b, z:= w * x
$$

### Backpropagation

$$
\frac{\partial L}{\partial L} = 1, \frac{\partial L}{\partial z} = 1, \frac{\partial L}{\partial b} = 1
$$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} * \frac{\partial z}{\partial w} = x
$$
