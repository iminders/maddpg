# -*- coding:utf-8 -*-
import numpy as np

from maddpg.common.logger import logger


def make_env(args=None, id=0):
    logger.info("create environment: %d" % id)
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
