# -*- coding:utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from maddpg.agents.base.agent import BaseAgent
from maddpg.common.logger import logger
from maddpg.nets.policy import get_policy_model
from maddpg.nets.value import get_value_model


class Agent(BaseAgent):
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        super(Agent, self).__init__(args, agent_num, act_spaces, obs_spaces)
        logger.info("policys:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.policys = self.create_policys()
        self.target_policys = self.create_policys()
        logger.info("values:act_shapes:%s, obs_shapes:%s" %
                    (str(self.act_shapes), str(self.obs_shapes)))
        self.values = self.create_values()
        self.target_values = self.create_values()

    def create_policys(self):
        policys = []
        for i in range(self.n):
            m = get_policy_model(
                i, self.args, self.act_shapes, self.obs_shapes)
            policys.append(m)
        return policys

    def create_values(self):
        values = []
        for i in range(self.n):
            m = get_value_model(i, self.args, self.act_shapes, self.obs_shapes)
            values.append(m)
        return values

    def add_graph(self):
        return None

    def action(self, obs):
        obs = np.asarray([obs])
        act = [self.policys[i].predict(obs)[0] for i in range(self.n)]
        return act

    def update_params(self, obs, act, rew, obs_next, done):
        # TODO
        time.sleep(1)
        [q_value, p_loss, q_loss, p_reg,
            act_reg] = np.random.random(5).tolist()
        return q_value, p_loss, q_loss, p_reg, act_reg
