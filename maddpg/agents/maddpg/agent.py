# -*- coding:utf-8 -*-

import gc
import time

import numpy as np

from maddpg.agents.base.agent import BaseAgent
from maddpg.common.logger import logger


class Agent(BaseAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.policys = []
        self.values = []
        self.target_policys = []
        self.target_values = []

    def action(self, obs):
        # TODO:
        return self.random_action()

    def update_params(self, obs, act, rew, obs_t, done):
        # TODO
        time.sleep(1)
        [q_value, p_loss, q_loss, p_reg,
            act_reg] = np.random.random(4).tolist()
        return q_value, p_loss, q_loss, p_reg, act_reg
