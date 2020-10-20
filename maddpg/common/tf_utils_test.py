# -*- coding:utf-8 -*-

import random
import sys

import numpy as np
import pytest
import tensorflow as tf

from maddpg.common.tf_utils import set_global_seeds


def test_set_global_seeds():
    set_global_seeds(0)
    actual = random.random()
    expected = 0.8444218515250481
    assert expected == actual

    actual = np.random.random([2, 2])
    expected = 0.5488135039273248
    assert expected == actual[0, 0]

    actual = tf.random.uniform(shape=()).numpy()
    expected = 0.29197514
    assert expected - actual < 1e-8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
