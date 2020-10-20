# -*- coding:utf-8 -*-

import sys

import pytest

from maddpg.arguments import parse_experiment_args
from maddpg.common.env_utils import get_act_shapes, get_obs_shapes, make_env


def test_get_shapes():
    args = parse_experiment_args()
    env = make_env(args=args, id=0)
    act_shapes = get_act_shapes(env)
    print(act_shapes)
    assert [2, 2, 2] == act_shapes
    obs_shapes = get_obs_shapes(env)
    print(obs_shapes)
    assert [4, 4, 4] == obs_shapes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
