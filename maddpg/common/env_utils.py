# -*- coding:utf-8 -*-
import numpy as np

from maddpg.common.logger import logger


def make_env(args=None, id=0):
    logger.debug("create environment: %d" % id)
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario_name = args.scenario
    benchmark = args.benchmark
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation)
    return env


def get_shapes(n_space):
    return [space.shape[0] for space in n_space]


def uniform_action(action_space):
    return [np.random.uniform(
        space.low, space.high, space.shape) for space in action_space]


def print_space_type(act_space):
    from gym import spaces
    from multiagent.multi_discrete import MultiDiscrete

    if isinstance(act_space, spaces.Box):
        assert len(act_space.shape) == 1
        print("Box:", act_space.shape[0])
    elif isinstance(act_space, spaces.Discrete):
        print("Discrete:", act_space.n)
    elif isinstance(act_space, MultiDiscrete):
        print("MultiDiscrete: low:", act_space.low, "high:", act_space.high)
    elif isinstance(act_space, spaces.MultiBinary):
        print("MultiBinary:", act_space.n)
    else:
        raise NotImplementedError
