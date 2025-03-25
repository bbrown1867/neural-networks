"""
Autograd (automatic gradient engine): Implements backpropagation.

What is backpropagation?
    Efficiently evaluate the gradient of a loss function with respect to the
    weights of a neural network. Allows us to iteratively tune the weights of
    the neural network to minimize the loss function.

var.data - Forward pass, evaluate the expression.
var.backward() - Backprop through the expression graph and apply chain rule.

Neural network is mathematical expression. Given weights and inputs, outputs are
predictions (or loss function).

Derivative of an output with respect to multiple inputs: Adjust one input
slightly, holding other ones constant. The derivative is how much the output
changed given the small change in a specific input.

Neural network: Multi-layer perceptron.

What is a neural network?
    Simple mathematical model of a neuron in the human brain: Inputs (x) and
    synapse (weights, w). The synapses are multiplied by the inputs. These
    input-weight products are summed up and a bias (b) is applied. Then the
    result is put into a non-linear activation function (e.g. tanh, relu).
"""

import math
import random

from graph import draw_graph


class Value:

    def __init__(self, data, children=(), op="", label=""):
        self.data = data
        self.prev = set(children)
        self.op = op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), op="+")

        # Derivative of x + a = 1
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), op="*")

        # Derivative of a*x = a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, children=(self,), op=f"**{other}")

        # Derivative of x**a = a*x**(a - 1)
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), children=(self,), op="exp")

        # Derivative of e**x = e**x
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def tanh(self):
        return ((2 * self).exp() - 1) / ((2 * self).exp() + 1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return self * (other**-1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


class Neuron:

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Implement the neuron: f(w*x + b), where f is the activation functon
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, num_inputs, num_neurons):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, num_inputs, layer_sizes):
        # Define the layers, based on the provided sizes plus the inputs
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def test_value():
    # d = a * b + c
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"

    # L = d * f
    f = Value(-2.0, label="f")
    L = d * f
    L.label = "L"

    # Test the forward pass
    assert L.data == -8.0

    draw_graph(L)


def test_neuron():
    # Inputs x1, x2
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # Weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # Bias
    b = Value(6.8813735870195432, label="b")

    # Neuron
    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x2w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1w1x2w2"
    n = x1w1x2w2 + b
    n.label = "n"

    # Activation function
    o = n.tanh()

    # Backprop
    o.backward()

    # Test backprop
    assert x1.grad == -1.5
    assert w1.grad == 1.0
    assert x2.grad == 0.5
    assert w2.grad == 0.0


def test_mlp():
    # Inputs
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    # Desired outputs
    ys = [1.0, -1.0, -1.0, 1.0]

    # Neural network definition
    n = MLP(3, [4, 4, 1])

    # Training parameters
    num_iterations = 100
    learning_rate = 0.05

    # Training loop: Gradient descent
    for k in range(num_iterations):
        # Forward pass
        yout = [n(x) for x in xs]

        # Loss function: Mean squared error
        loss = sum([(y1 - y2) ** 2 for y1, y2 in zip(ys, yout)])

        # Backprop
        for p in n.parameters():
            p.grad = 0.0
        loss.backward()

        # Update
        for p in n.parameters():
            # Note: Negate to decrease loss
            p.data += -1 * learning_rate * p.grad

        print(f"{k}: Loss = {loss.data}")

    for y1, y2 in zip(ys, yout):
        print(f"Expected: {y1}, Actual: {y2.data}")


if __name__ == "__main__":
    test_value()
    test_neuron()
    test_mlp()
