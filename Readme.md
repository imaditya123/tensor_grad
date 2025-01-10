# UniTensor and Neural Network Library

This project implements a lightweight, autograd-enabled tensor class `UniTensor` and a modular neural network framework consisting of `Neuron`, `Layer`, and `MLP` classes. It is designed for educational purposes, demonstrating the core principles of automatic differentiation and modular neural network design.

---

## Features

### `UniTensor`
- **Scalar Tensor Operations**: Supports addition, subtraction, multiplication, division, and power operations.
- **Activation Functions**: Includes `tanh`, `relu`, and `exp`.
- **Backward Propagation**: Performs automatic differentiation using a reverse-mode gradient computation.
- **Visualization**: Generates computational graphs with `graphviz`.

### Neural Network Modules
- **Modular Design**: Consists of `Neuron`, `Layer`, and `MLP` (multi-layer perceptron).
- **Custom Activation**: Supports ReLU and linear activation for neurons.
- **Gradient Reset**: Built-in utility to reset gradients.
- **Parameter Access**: Easily retrieve all learnable parameters for optimization.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. Install dependencies:
   ```bash
   pip install graphviz
   ```
3. (Optional) Install additional tools for working with Python projects.

---

## File Structure
- **`tensor_autograd.py`**: Defines the `UniTensor` class, implementing basic operations, autograd functionality, and computational graph visualization.
- **`nn.py`**: Contains `Module`, `Neuron`, `Layer`, and `MLP` classes for building neural networks.

---

## Usage

### Using `UniTensor`
```python
from grad import UniTensor

# Create tensors
a = UniTensor(2.0)
b = UniTensor(3.0)

# Perform operations
c = a * b + b ** 2
c.label = "Output"

# Backpropagation
c.backward()

# Print gradients
print(a, b, c)

# Visualize the computational graph
c.draw_dot().render("comp_graph", format="png")
```

### Building and Using a Neural Network
```python
from nn import MLP
from grad import UniTensor
import random

# Define a network
mlp = MLP(nin=3, nouts=[4, 4, 1])
print(mlp)

# Input data
x = [UniTensor(random.uniform(-1, 1)) for _ in range(3)]

# Forward pass
output = mlp(x)

# Set a loss (mean squared error example)
loss = (output - UniTensor(1.0)) ** 2

# Backward pass
loss.backward()

# Access gradients
for param in mlp.parameters():
    print(param)

# Reset gradients
mlp.zero_grad()
```

---

## Class Descriptions

### `UniTensor`
A scalar tensor class with support for:
- Arithmetic operations: `+`, `-`, `*`, `/`, `**`
- Unary operations: `-` (negation), `exp`, `tanh`, `relu`
- Backward propagation: Computes gradients via reverse-mode autodiff
- Computational graph visualization: Generates a DOT file to visualize the graph

### `Module`
Abstract base class for neural network components. Implements utility functions like `zero_grad` and `parameters`.

### `Neuron`
Represents a single neuron with:
- Learnable weights and bias
- Optional ReLU activation

### `Layer`
A collection of `Neuron` objects.

### `MLP`
A multi-layer perceptron built from a sequence of `Layer` objects.

---

## Examples

### Visualizing a Computational Graph
```python
from grad import UniTensor

a = UniTensor(1.0)
b = UniTensor(2.0)
c = a + b * b

c.backward()
c.draw_dot(rankdir='TB').render("graph", format="png")
```

### Building a Simple Neural Network
```python
from nn import MLP
from grad import UniTensor
import random

# Initialize an MLP with 2 input neurons, one hidden layer of 3 neurons, and 1 output neuron
model = MLP(nin=2, nouts=[3, 1])

# Input data
data = [UniTensor(random.uniform(-1, 1)) for _ in range(2)]

# Forward pass
output = model(data)
print(output)

# Compute loss and backpropagate
loss = (output - UniTensor(1.0)) ** 2
loss.backward()

# View parameters and their gradients
for param in model.parameters():
    print(param)

# Reset gradients
model.zero_grad()
```

---

## Requirements
- Python 3.7+
- `graphviz` (install using your package manager or via `pip` for Python bindings)

---

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute as needed.

