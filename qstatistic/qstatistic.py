# -*- coding: utf-8 -*-
import math


class qExponential:
    def __init__(self, x, q):
        self.x = x
        self.q = q

    def calc(self):
        if self.q == 1:
            return math.exp(self.x)
        else:
            temp = 1 + (1 - self.q)*self.x
            if temp > 0:
                return math.pow(temp, 1/(1 - self.q))
            else:
                return 0


class qLogarithm:
    def __init__(self, x, q):
        self.x = x
        self.q = q

    def calc(self):
        if self.x >= 0:
            if self.q == 1:
                return math.log(self.x)
            else:
                return (math.pow(self.x, 1 - self.q) - 1) / (1 - self.q)
        else:
            raise ValueError('Undefined')


class qGaussian:
    def __init__(self, x, B, q):
        self.x = x
        self.B = B
        self.q = q

        if not type(q) == int and not type(q) == float:
            self.q = 1
            raise ValueError('q must be an Integer or a Float')

        # q exponential function e^-Bx²
        self.e = qExponential(
            x=-self.B*math.pow(self.x, 2),
            q=self.q
        ).calc()

        # Normalization factor Cq
        if self.q == 1:
            self.Cq = math.sqrt(math.pi)
        else:
            if self.q < 1:
                self.Cq = 2*math.sqrt(math.pi)*math.gamma(1/(1-q))*(1/((3-q)*math.sqrt(1-q)*math.gamma((3-q)/(2*(1-q)))))
            else:
                self.Cq = math.sqrt(math.pi)*math.gamma((3-q)/(2*(q-1)))*(1/(math.sqrt(q-1)*math.gamma(1/(q-1))))

    def calc(self):
        return (math.sqrt(self.B)/self.Cq)*self.e


# class qEntropy:

class qSigmoide:
    def __init__(self, L, I, B, a, q):
        self.L = L
        self.I = I
        self.B = B
        self.a = a
        self.q = q

        self.D = math.fabs((self.I - self.B)/self.a) + 0.0001


    def calc(self):

        if self.q == 1:
            result = self.L - (self.L / (1 + math.exp(-self.D)))
        else:
            if (1 + (1 - self.q)*self.D) < 0:
                self.D = -1 / self.D
            result = self.L / math.pow(1 + (1 - self.q)*self.D, 1 / (1 - self.q))

        if result <= 0:
            result = 0

        if result == float('inf'):
            result = self.L

        return [result, self.D]


class qWeibull:
    def __init__(self, x, q, lamb, k):
        self.x = x
        self.q = q
        self.lamb = lamb
        self.k = k

    def calc(self):
        if self.x >= 0:
            return (2 - self.q)*(self.k/self.lamb)*math.pow(self.x/self.lamb, self.k - 1) * \
                            qExponential(x=-math.pow(self.x/self.lamb, self.k), q=self.q).calc()
        else:
            return 0
