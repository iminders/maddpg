# -*- coding:utf-8 -*-
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


def get_act_shapes(env):
    shapes = []
    logger.info(str(env.action_space))
    for space in env.action_space:
        shapes.append(space.shape[0])
    return shapes


def get_obs_shapes(env):
    return [env.observation_space[i].shape[0] for i in range(env.n)]
