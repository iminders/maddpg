# -*- coding:utf-8 -*-

import numpy as np

from maddpg.common.env_utils import get_shapes
from maddpg.common.logger import logger


class Agent(object):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.act_spaces = act_spaces
        self.obs_spaces = obs_spaces
        self.act_shapes = get_shapes(act_spaces)
        self.obs_shapes = get_shapes(obs_spaces)
        logger.info("init agent, act_shapes=%s, obs_shapes=%s" (
            self.act_shapes, self.obs_shapes))
        self.n = agent_num
        self.policys = []
        self.values = []
        self.target_policys = []
        self.target_values = []

    def action(self, obs):
        return self.random_action()

    def random_action(self):
        action = [np.random.uniform(self.act_shapes[i]) for i in range(self.n)]
        return action

    def serve():
        logger.info()
