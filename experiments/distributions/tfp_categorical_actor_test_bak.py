# -*- coding:utf-8 -*-

import sys

import numpy as np
import pytest
import tensorflow as tf

from maddpg.common.tf_utils import set_global_seeds
from tfp_categorical_actor import CategoricalActorCritic


class TestTFPCategoricalActor(tf.test.TestCase):
    def test_sample(self):
        set_global_seeds(0)
        pd = CategoricalActorCritic(state_shape=(5,), action_dim=5)
        action, log_prob, v = pd(np.random.random((10, 5)))
        # print(action, log_prob, v)
        action = pd(np.random.random((10, 5)), test=True)
        print(action)
        expected = np.asarray([])
        self.assertAllEqual(action, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
