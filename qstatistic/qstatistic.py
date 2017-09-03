# -*- coding: utf-8 -*-
import math
import numpy as np


class qExponential:
    def __init__(self, x, q):
        self.x = x
        self.q = q

    def calc(self):
        if self.q == 1:
            return np.exp(self.x)
        else:
            temp = 1 + (1 - self.q)*self.x
            if temp > 0:
                return np.power(temp, 1/(1 - self.q))
            else:
                return 0


class qLogarithm:
    def __init__(self, x, q):
        self.x = x
        self.q = q

    def calc(self):
        if self.x >= 0:
            if self.q == 1:
                return np.log(self.x)
            else:
                return (np.power(self.x, 1 - self.q) - 1) / (1 - self.q)
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
            x=-self.B*np.power(self.x, 2),
            q=self.q
        ).calc()

        # Normalization factor Cq
        if self.q == 1:
            self.Cq = np.sqrt(np.pi)
        else:
            if self.q < 1:
                self.Cq = 2*np.sqrt(np.pi)*math.gamma(1/(1-q))*(1/((3-q)*np.sqrt(1-q)*math.gamma((3-q)/(2*(1-q)))))
            else:
                self.Cq = np.sqrt(np.pi)*math.gamma((3-q)/(2*(q-1)))*(1/(np.sqrt(q-1)*math.gamma(1/(q-1))))

    def calc(self):
        return (np.sqrt(self.B)/self.Cq)*self.e


# class qEntropy:

class qSigmoid:
    def __init__(self, I, L, B, a, q):
        """
        Sigmoid Extended Function

        :param I: Input Value
        :param L: Maximum luminance value
        :param B: Luminance value sought in the application
        :param a: Range of luminance values around B
        :param q:
        """
        self.L = L
        self.I = I
        self.B = B
        self.a = a
        self.q = q

        self.D = np.absolute((self.I - self.B)/self.a) + 0.0001

        # In case of 'q' equals one, the result will be the traditional sigmoid function.
        # Otherwise, we must invert D if (1 + (1 - self.q) * self.D < 0).
        if self.q != 1:
            if type(self.D) == np.ndarray:
                # In case of input be an array invert only the points that satisfy the condition
                self.D[(1 + (1 - self.q) * self.D) < 0] = -1 / self.D[(1 + (1 - self.q) * self.D) < 0]
            elif type(self.D) == int or type(self.D) == float:
                if (1 + (1 - self.q) * self.D) < 0:
                    self.D = -1 / self.D

    def get_D(self):
        return self.D

    def calc(self):
        if self.q == 1:
            result = self.L - (self.L / (1 + np.exp(-self.D)))
        else:
            result = self.L / np.power(1 + (1 - self.q)*self.D, 1 / (1 - self.q))

        # If result is < 0 or infinity then we must set to 0 or L value.
        if type(result) == np.ndarray:
            result[result < 0] = 0
            result[result == float('inf')] = self.L
        elif result == int or result == float:
            result = 0 if result < 0 else result
            result = self.L if result == float('inf') else result

        return result


class qWeibull:
    def __init__(self, x, q, lamb, k):
        self.x = x
        self.q = q
        self.lamb = lamb
        self.k = k

    def calc(self):
        if self.x >= 0:
            return (2 - self.q)*(self.k/self.lamb)*np.power(self.x/self.lamb, self.k - 1) * \
                            qExponential(x=-np.power(self.x/self.lamb, self.k), q=self.q).calc()
        else:
            return 0
