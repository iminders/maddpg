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
    i, episode, in_episode_step = 0, 0, 0
    episode_rews = [0] * args.save_rate

    while True:
        next_obs, rew, done, info = env.step(action)
        i += 1
        in_episode_step += 1
        terminal = (in_episode_step >= args.max_episode_len)
        sample = merge_sample(obs, action, next_obs, rew, done, terminal)
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
        if str(action) == "stop":
            logger.info("[%d],%d finished explore, learning server stoped" % (
                id, i))
            break
        obs = next_obs
        if all(done) or terminal:
            obs = env.reset()
            in_episode_step = 0
            episode += 1
            episode_rews[episode % args.save_rate] = np.mean(rew)
            if episode % args.save_rate == 0:
                episode_avg_rews = np.mean(episode_rews)
                logger.debug(
                    "env[%d] step:%d, episode:%d episode avg rew: %.5f" %
                    (id, i, episode, episode_avg_rews))


def parallel_explore(args):
    processes = []
    for i in range(args.num_env):
        p = Process(target=explore, args=(args, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def merge_sample(obs, action, next_obs, rew, done, terminal):
    return [np.concatenate(obs),
            np.concatenate(action),
            np.concatenate(next_obs),
            rew, done, terminal]
