# neural-network

### Weighted Sum

- $a$ = input/activation
- $W$ = weight
- $B$ = bias

where $a$ is a transposed vector $|a|^T$

$z(x) = W \cdot a + B$

### Activation function (sigmoid)

$\sigma(z) = 1 / 1 + e^{-z}$

### Activation function derivative (sigmoid prime)

$\sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))$

### Cost 

- $y$ = predicted output
- $\hat y$ = expected output

$C(y) = (y - \hat y)^2$


### Cost derivative

$C'(y) = 2(y - \hat y)$



### Back Propagation

#### Weighted Sum Partial derivative with respect to w

$\frac{\partial z}{\partial W}(W \cdot a + B) = a^T$

$\frac{\partial z}{\partial W} = a^T$

#### Weighted Sum Partial derivative with respect to b

$\frac{\partial z}{\partial b}(W \cdot a + B) = 1$

$\frac{\partial z}{\partial B} = 1$

#### Error term

$\sigma^{\prime} =$ sigmoid prime

$\delta =$ error term

$l =$ layer

$z = W \cdot a + B$ 

$W = $
weights matrix 

$C^{\prime} = $ 
cost derivative

if $n = l$ (output layer):

$\delta^{[l]} = C^{\prime}(a^{[n]}) \cdot \sigma^{\prime}(z^{[n]})$ 

if $l < n$ (hidden layers):

$\delta^{[l]} = (W^{[l+1]})^T \cdot \delta^{[l+1]} \cdot \sigma^{\prime}(z^{[l]})$


#### weight gradient

partial derivative of z with respect to W

$\frac{\partial z}{\partial W^{[l]}} = (a^{[l-1]})^T$

$\nabla W^{[l]} = \delta^{[l]} \cdot \frac{\partial z}{\partial W^{[l]}}$

which simplifies to 

$\nabla W^{[l]} = \delta^{[l]} \cdot (a^{[l-1]})^T$

#### bias gradient

partial derivative of z with respect to B

$\frac{\partial z}{\partial B^{[l]}} = 1$

$\nabla B^{[l]} = \delta^{[l]}$

#### adjusting weight and bias
- $lr$ = learning rate
- $\nabla w$ = weight gradient
- $\nabla b$ = bias gradient

$\Delta w = lr \cdot \nabla w$

$w = w - \Delta w$

$\Delta b = lr \cdot \nabla b$

$b = b - \Delta b$

