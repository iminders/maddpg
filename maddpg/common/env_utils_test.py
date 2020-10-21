# -*- coding:utf-8 -*-

import sys

import pytest

from maddpg.arguments import parse_experiment_args
from maddpg.common.env_utils import get_shapes, make_env


def test_get_shapes():
    args = parse_experiment_args()
    env = make_env(args=args, id=0)
    act_shapes = get_shapes(env.action_space)
    assert [2, 2, 2] == act_shapes
    obs_shapes = get_shapes(env.observation_space)
    assert [4, 4, 4] == obs_shapes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
