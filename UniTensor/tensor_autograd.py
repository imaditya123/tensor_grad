import math
from graphviz import Digraph

class UniTensor:
    """
    UniTensor: A class representing a single-value tensor supporting basic operations
    and automatic differentiation.
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialize the UniTensor object.

        Args:
            data (float/int): The scalar value of the tensor.
            _children (tuple): Parent tensors involved in creating this tensor.
            _op (str): Operation that created this tensor.
        """
        self.data = data
        self.grad = 0  # Gradient for backpropagation.
        self.label = ""  # Optional label for identification.
        self._prev = set(_children)  # Parents of this tensor.
        self._op = _op  # Operation that produced this tensor.
        self._backward = lambda: None  # Backward function for gradient computation.

    def __repr__(self):
        """
        String representation of the UniTensor object.
        """
        return f"Label:{self.label}|Data:{self.data}|Grad:{self.grad}"

    def __add__(self, other):
        """
        Addition of two UniTensors (or UniTensor with scalar).

        Args:
            other (UniTensor/int/float): The tensor or scalar to add.

        Returns:
            UniTensor: Result of the addition.
        """
        other = other if isinstance(other, UniTensor) else UniTensor(other)
        out = self.data + other.data

        output = UniTensor(out, (self, other), _op='+')

        def backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = backward
        return output

    def __mul__(self, other):
        """
        Multiplication of two UniTensors (or UniTensor with scalar).

        Args:
            other (UniTensor/int/float): The tensor or scalar to multiply.

        Returns:
            UniTensor: Result of the multiplication.
        """
        other = other if isinstance(other, UniTensor) else UniTensor(other)
        out = self.data * other.data
        output = UniTensor(out, (self, other), _op='*')

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = backward
        return output

    def __pow__(self, power):
        """
        Power operation on a UniTensor.

        Args:
            power (int/float): The exponent to raise the tensor's value to.

        Returns:
            UniTensor: Result of the power operation.
        """
        assert isinstance(power, (int, float)), "Only supported for int/float power for now"
        out = self.data ** power
        output = UniTensor(out, (self,), _op=f'**{power}')

        def backward():
            self.grad += power * (self.data ** (power - 1)) * output.grad
        output._backward = backward
        return output

    def __neg__(self):
        """
        Negation of a UniTensor.

        Returns:
            UniTensor: Result of the negation.
        """
        out = self.data * -1
        output = UniTensor(out, (self,), _op=f'-{self.data}')

        def backward():
            self.grad += -1 * output.grad
        output._backward = backward
        return output

    def exp(self):
        """
        Exponential function applied to the tensor.

        Returns:
            UniTensor: Result of the exponential function.
        """
        out = math.exp(self.data)
        output = UniTensor(out, (self,), _op=f'e^{out}')

        def backward():
            self.grad += out * output.grad
        output._backward = backward
        return output

    def backward(self):
        """
        Perform backpropagation to compute gradients.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1  # Seed gradient for backpropagation.
        for v in reversed(topo):
            v._backward()

    def tanh(self):
        """
        Hyperbolic tangent activation function.

        Returns:
            UniTensor: Result of the tanh function.
        """
        out = math.tanh(self.data)
        output = UniTensor(out, (self,), _op=f'tanh({out})')

        def backward():
            self.grad += (1 - out ** 2) * output.grad
        output._backward = backward
        return output

    def relu(self):
        """
        Rectified Linear Unit (ReLU) activation function.

        Returns:
            UniTensor: Result of the ReLU function.
        """
        out = UniTensor(0 if self.data < 0 else self.data, (self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        """
        Subtraction of two UniTensors (or UniTensor with scalar).

        Args:
            other (UniTensor/int/float): The tensor or scalar to subtract.

        Returns:
            UniTensor: Result of the subtraction.
        """
        return self + (-other)

    def __truediv__(self, other):
        """
        Division of two UniTensors (or UniTensor with scalar).

        Args:
            other (UniTensor/int/float): The tensor or scalar to divide.

        Returns:
            UniTensor: Result of the division.
        """
        return self * (other ** -1)

    def __radd__(self, other):
        """
        Reverse addition (other + self).

        Args:
            other (UniTensor/int/float): The tensor or scalar to add.

        Returns:
            UniTensor: Result of the addition.
        """
        return self + other

    def __rmul__(self, other):
        """
        Reverse multiplication (other * self).

        Args:
            other (UniTensor/int/float): The tensor or scalar to multiply.

        Returns:
            UniTensor: Result of the multiplication.
        """
        return self * other

    def __rsub__(self, other):
        """
        Reverse subtraction (other - self).

        Args:
            other (UniTensor/int/float): The tensor or scalar to subtract.

        Returns:
            UniTensor: Result of the subtraction.
        """
        return other + (-self)

    def __rtruediv__(self, other):
        """
        Reverse division (other / self).

        Args:
            other (UniTensor/int/float): The tensor or scalar to divide.

        Returns:
            UniTensor: Result of the division.
        """
        return other * self ** -1

    

    def draw_dot(self, format='svg', rankdir='LR'):
        def trace(root):
            nodes, edges = set(), set()
            def build(v):
                if v not in nodes:
                    nodes.add(v)
                    for child in v._prev:
                        edges.add((child, v))
                        build(child)
            build(root)
            return nodes, edges
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        root=self
        assert rankdir in ['LR', 'TB']
        nodes, edges = trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
        
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
