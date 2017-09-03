#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `qstatistic` package."""


import unittest
import numpy as np
from qstatistic import qstatistic as qs


class TestQstatistic(unittest.TestCase):
    """Tests for `qstatistic` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_qSigmoid_point(self):
        result = qs.qSigmoid(
            L=255,
            I=128,
            B=128,
            a=15,
            q=0.35
        ).calc()

        self.assertEqual(result, 254.97450210358875)

    def test_qSigmoid_1d(self):
        array = np.round(np.random.rand(10)*255)
        L = np.max(array)
        B = L/2
        a = (L-B)/4
        q = 0.35

        result = qs.qSigmoid(L=L, I=array, B=B, a=a, q=q).calc()

        for idx in range(array.shape[0]):
            self.assertEqual(result[idx], qs.qSigmoid(L=L, I=array[idx], B=B, a=a, q=q).calc())

        self.assertEqual(array.shape, result.shape)

    def test_qSigmoid_2d(self):
        array = np.round(np.random.rand(10, 10)*255)
        L = np.max(array)
        B = L/2
        a = (L-B)/4
        q = 0.35

        result = qs.qSigmoid(L=L, I=array, B=B, a=a, q=q).calc()

        print(result.shape)

        for lin in range(array.shape[0]):
            for col in range(array.shape[1]):
                self.assertEqual(result[lin, col], qs.qSigmoid(L=L, I=array[lin, col], B=B, a=a, q=q).calc())

        self.assertEqual(array.shape, result.shape)

        print(result.shape)
