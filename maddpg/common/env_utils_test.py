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
        assert [2, 2, 2] == act_shapes
        obs_shapes = get_shapes(self.env.observation_space)
        assert [4, 4, 4] == obs_shapes

    def test_uniform_action(self):
        actions = uniform_action(self.env.action_space)
        actions = [a.tolist() for a in actions]
        print(actions[0])
        print(actions[1])
        print(actions[2])
        assert [0.0976270078546495, 0.43037873274483895] == actions[0]
        assert [0.20552675214328775, 0.08976636599379373] == actions[1]
        assert [-0.15269040132219058, 0.29178822613331223] == actions[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
