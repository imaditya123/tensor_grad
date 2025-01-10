import unittest
import torch
from math import exp, tanh
from UniTensor.tensor_autograd import UniTensor

class TestUniTensor(unittest.TestCase):
    #  check initialization
    def test_initialization(self):
        t = UniTensor(5)
        self.assertEqual(t.data, 5)
        self.assertEqual(t.grad, 0)
        self.assertEqual(t.label, "")
        self.assertEqual(t._op, "")
    # Checking different operation
    def test_addition(self):
        t1 = UniTensor(3)
        t2 = UniTensor(4)
        result = t1 + t2
        self.assertEqual(result.data, 7)

        result.backward()
        self.assertEqual(t1.grad, 1)
        self.assertEqual(t2.grad, 1)

    def test_multiplication(self):
        t1 = UniTensor(3)
        t2 = UniTensor(4)
        result = t1 * t2
        self.assertEqual(result.data, 12)

        result.backward()
        self.assertEqual(t1.grad, 4)
        self.assertEqual(t2.grad, 3)

    def test_power(self):
        t = UniTensor(2)
        result = t ** 3
        self.assertEqual(result.data, 8)

        result.backward()
        self.assertEqual(t.grad, 12)  # 3 * (2^2)

    def test_negation(self):
        t = UniTensor(5)
        result = -t
        self.assertEqual(result.data, -5)

        result.backward()
        self.assertEqual(t.grad, -1)

    def test_exponential(self):
        t = UniTensor(1)
        result = t.exp()
        self.assertAlmostEqual(result.data, exp(1), places=5)

        result.backward()
        self.assertAlmostEqual(t.grad, exp(1), places=5)

    def test_tanh(self):
        t = UniTensor(0.5)
        result = t.tanh()
        self.assertAlmostEqual(result.data, tanh(0.5), places=5)

        result.backward()
        self.assertAlmostEqual(t.grad, 1 - tanh(0.5)**2, places=5)

    def test_relu(self):
        t1 = UniTensor(2)
        t2 = UniTensor(-1)

        result1 = t1.relu()
        self.assertEqual(result1.data, 2)

        result2 = t2.relu()
        self.assertEqual(result2.data, 0)

        result1.backward()
        result2.backward()
        self.assertEqual(t1.grad, 1)
        self.assertEqual(t2.grad, 0)

    def test_subtraction(self):
        t1 = UniTensor(5)
        t2 = UniTensor(3)
        result = t1 - t2
        self.assertEqual(result.data, 2)

        result.backward()
        self.assertEqual(t1.grad, 1)
        self.assertEqual(t2.grad, -1)

    def test_division(self):
        t1 = UniTensor(6)
        t2 = UniTensor(3)
        result = t1 / t2
        self.assertEqual(result.data, 2)

        result.backward()
        self.assertEqual(t1.grad, 1/3)
        self.assertEqual(t2.grad, -6/(3**2))

    def test_reverse_operations(self):
        t = UniTensor(5)
        result1 = 3 + t
        result2 = 3 * t
        result3 = 3 - t
        result4 = 3 / t

        self.assertEqual(result1.data, 8)
        self.assertEqual(result2.data, 15)
        self.assertEqual(result3.data, -2)
        self.assertAlmostEqual(result4.data, 3/5, places=5)

        result4.backward()
        self.assertAlmostEqual(t.grad, -3/(5**2), places=5)
    
    # Checking the forward and backpropagation 
    def test_sanity_check(self):

        x = UniTensor(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xmg, ymg = x, y

        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        xpt, ypt = x, y

        # forward pass went well
        self.assertAlmostEqual(ymg.data ,ypt.data.item(),places=5)
        # backward pass went well
        self.assertAlmostEqual(xmg.grad ,xpt.grad.item(),places=5)

    def test_more_ops(self):

        a = UniTensor(-4.0)
        b = UniTensor(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6

        # forward pass went well
        self.assertTrue( abs(gmg.data - gpt.data.item()) < tol)
        # backward pass went well
        self.assertTrue(abs(amg.grad - apt.grad.item()) < tol)
        self.assertTrue(abs(bmg.grad - bpt.grad.item()) < tol)



if __name__ == '__main__':
    unittest.main()
