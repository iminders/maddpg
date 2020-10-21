# -*- coding:utf-8 -*-

import os

from tensorflow.summary import SummaryWriter

from maddpg.common.env_utils import get_shapes, uniform_action
from maddpg.common.logger import logger
from maddpg.common.storage import Storage


class BaseAgent:
    def __init__(self, args, agent_num, act_spaces, obs_spaces):
        self.args = args
        self.act_spaces = act_spaces
        self.obs_spaces = obs_spaces
        self.act_shapes = get_shapes(act_spaces)
        self.obs_shapes = get_shapes(obs_spaces)
        logger.info("init agent, act_shapes=%s, obs_shapes=%s" (
            self.act_shapes, self.obs_shapes))
        self.n = agent_num
        # 初始化目录
        self.tb_dir = self.must_get_dir(os.path.join(
            args.tb_dir, args.runner, args.run_id))
        self.model_dir = self.must_get_dir(os.path.join(
            args.model_dir, args.runner, args.run_id))
        self.writer = SummaryWriter(self.tb_dir)
        # s3云存储
        self.stoarge = Storage(args)

    def random_action(self):
        return uniform_action(self.act_spaces)

    def action(self):
        raise NotImplementedError

    def learn(self, iter):
        raise NotImplementedError

    def must_get_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
