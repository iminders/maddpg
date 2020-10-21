# -*- coding:utf-8 -*-

import numpy as np

from maddpg.common.logger import logger


class Agent(object):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        logger.info("init agent, act_shapes=%s, obs_shapes=%s" (
            act_shapes, obs_shapes))
        self.n = agent_num
        self.act_shapes = act_shapes
        self.obs_shapes = act_shapes
        self.policys = []
        self.values = []
        self.target_policys = []
        self.target_values = []

    def action(self, obs):
        return None

    def random_action(self):
        action = [[np.random.random()] for i in range(self.n)]

    def serve():
        logger.info()
