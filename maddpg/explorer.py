# -*- coding:utf-8 -*-
import pickle
import time
import zlib
from multiprocessing import Process

import zmq

from maddpg.common.env_wrappers import BatchedEnvironment
from maddpg.common.logger import logger


def increment(items, size):
    for i in range(size):
        items[i] += 1


def explore(args, id):
    c = zmq.Context()
    s = c.socket(zmq.REQ)
    host = 'tcp://%s:%d' % (args.host, args.port)
    s.connect(host)
    logger.info('zmq socket addr: tcp://%s:%d' % (args.host, args.port))
    batch_env = BatchedEnvironment(args, id)
    obs = batch_env.reset()
    action = batch_env.uniform_action()
    i = 0
    n = args.env_batch_size
    episode = [0] * n
    episode_step = [0] * n

    while True:
        next_obs, rew, done, info = batch_env.step(action)
        i += n
        increment(episode_step, n)
        terminal = [episode_step[i] >= args.max_episode_len for i in range(n)]
        sample = [obs, action, next_obs, rew, done, terminal]
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

        if i % (10 * args.save_rate) == 0:
            logger.debug("batch_env[%d] step:%i, episode:%s" %
                         (id, i, str(episode)))
        obs = batch_env.reset_if_done(done, terminal, episode_step, episode)
        if i % 10000 == 0:
            logger.debug(str(id) + ":" + str(episode))


def parallel_explore(args):
    processes = []
    for i in range(args.num_env):
        p = Process(target=explore, args=(args, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
