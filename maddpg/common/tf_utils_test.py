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

    actual = np.random.random([2, 2]).tolist()
    expected = [[0.5488135039273248, 0.7151893663724195],
                [0.6027633760716439, 0.5448831829968969]]
    assert expected == actual

    actual = tf.random.uniform(shape=(2, ), dtype=tf.float32).numpy().tolist()
    expected = [0.29197514057159424, 0.2065664529800415]
    assert expected == actual


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
