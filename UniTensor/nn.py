from UniTensor.tensor_autograd import UniTensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self,nin,nonlin=True):
        self.w={UniTensor(random.uniform(-1,1)) for _ in range(nin)}
        self.b=UniTensor(0)
        self.nonlin=nonlin

    def __call__(self,x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w+[self.b]

    def __repr__(self,):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self,nin,nouts):
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]

    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters() ]

    def __repr__(self):
        return f"Layer of [{','.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1],nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
