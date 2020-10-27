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
        assert [5, 5, 5] == act_shapes
        obs_shapes = get_shapes(self.env.observation_space)
        assert [4, 4, 4] == obs_shapes

    def test_uniform_action(self):
        actions = uniform_action(self.env.action_space)
        actions = [a.tolist() for a in actions]
        assert [0.5488135039273248, 0.7151893663724195, 0.6027633760716439,
                0.5448831829968969, 0.4236547993389047] == actions[0]
        assert [0.6458941130666561, 0.4375872112626925, 0.8917730007820798,
                0.9636627605010293, 0.3834415188257777] == actions[1]
        assert [0.7917250380826646, 0.5288949197529045, 0.5680445610939323,
                0.925596638292661, 0.07103605819788694] == actions[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
