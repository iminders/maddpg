# -*- coding:utf-8 -*-

import sys

import pytest

from maddpg.arguments import parse_experiment_args
from maddpg.common.env_utils import get_shapes, make_env, uniform_action
from maddpg.common.tf_utils import set_global_seeds


class TestEnvUtils:
    def setup(self):
        args = parse_experiment_args()
        self.env = make_env(args=args, id=0)
        set_global_seeds(0)

    def test_get_shapes(self):
        act_shapes = get_shapes(self.env.action_space)
        from gym import spaces
        assert isinstance(self.env.action_space[0], spaces.Discrete)
        assert [5, 5, 5] == act_shapes
        obs_shapes = get_shapes(self.env.observation_space)
        assert [4, 4, 4] == obs_shapes

    def test_uniform_action(self):
        actions = uniform_action(self.env.action_space)
        actions = [a.tolist() for a in actions]
        assert [0.0976270078546495, 0.43037873274483895, 0.20552675214328775,
                0.08976636599379373, -0.15269040132219058] == actions[0]
        assert [0.29178822613331223, -0.12482557747461498, 0.7835460015641595,
                0.9273255210020586, -0.2331169623484446] == actions[1]
        assert [0.5834500761653292, 0.05778983950580896, 0.13608912218786462,
                0.8511932765853221, -0.8579278836042261] == actions[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
