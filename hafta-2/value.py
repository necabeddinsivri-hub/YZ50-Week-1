# Value sinifi - kendi mini autograd motorum
# YZ50 Hafta 2

import math


class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # toplamada turev 1, gradient aynen gecer
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # carpmada turev, digerinin degeri
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), 'pow')

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # tanh'i parcalara ayirmis hali (Gorev 4)
    def tanh2(self):
        e = (self * 2).exp()
        return (e + (-1)) * ((e + 1) ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def backward(self):
        # once dugumleri dogru siraya diz
        siralama = []
        gorulen = set()

        def sirala(v):
            if v not in gorulen:
                gorulen.add(v)
                for cocuk in v._prev:
                    sirala(cocuk)
                siralama.append(v)

        sirala(self)

        # sonra tersten gez
        self.grad = 1.0
        for dugum in reversed(siralama):
            dugum._backward()
