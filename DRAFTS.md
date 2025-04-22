#### Forward pass

$$
L = w * x + b, z:= w * x
$$

#### Backpropagation

$$
\frac{\partial L}{\partial L} = 1, \frac{\partial L}{\partial z} = 1, \frac{\partial L}{\partial b} = 1
$$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} * \frac{\partial z}{\partial w} = x
$$
