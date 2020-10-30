# -*- coding:utf-8 -*-

import sys

import numpy as np
import pytest
import tensorflow as tf

from maddpg.common.tf_utils import set_global_seeds
from maddpg.nets.mlp import MLP


class TestMLP(tf.test.TestCase):
    def test_mlp(self):
        set_global_seeds(0)
        num_units, input_size, output_size = 64, 10, 2
        m = MLP(num_units, input_size, output_size)
        x = np.random.random((2, 10)).astype(np.float32)
        actual = m(x).numpy()
        expected = np.asarray([[0.17461237, 0.14123951],
                               [0.03389007, 0.20079815]], dtype=np.float32)
        self.assertAllEqual(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
