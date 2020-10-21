# -*- coding:utf-8 -*-
import pickle
import time
import zlib
from multiprocessing import Process

import numpy as np
import zmq

from maddpg.common.env_utils import make_env, uniform_action
from maddpg.common.logger import logger


def explore(args, id):
    c = zmq.Context()
    s = c.socket(zmq.REQ)
    host = 'tcp://%s:%d' % (args.host, args.port)
    s.connect(host)
    logger.info('zmq socket addr: tcp://%s:%d' % (args.host, args.port))
    env = make_env(args, id)
    obs = env.reset()
    action = uniform_action(env.action_space)
    i = 0
    while True:
        next_obs, reward, done, info = env.step(action)
        sample = [obs, action, next_obs, reward, done, id]
        p = pickle.dumps(sample)
        z = zlib.compress(p)
        while True:
            try:
                s.send_pyobj(z)
                data = s.recv_pyobj()
                action = pickle.loads(data)
                break
            except zmq.ZMQError:
                logger.error("send to zmq server[%s] error, sleep 1s" % host)
                time.sleep(1)
        i += 1
        if str(action) == "stop":
            logger.info("[%d],%d finished explore, learning server stoped" % (
                id, i))
            break
        obs = next_obs
        if done:
            logger.info("env[%d] %d episode reward: %s, mean: %.5f" %
                        (id, i, str(reward), np.mean(reward)))
            obs = env.reset()


def parallel_explore(args):
    processes = []
    for i in range(args.num_env):
        p = Process(target=explore, args=(args, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
