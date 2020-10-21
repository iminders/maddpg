# -*- coding:utf-8 -*-
import pickle
import time
import zlib

import numpy as np
import tensorflow as tf
import zmq

from maddpg.agents.maddpg.agent import Agent
from maddpg.common.logger import logger


def make_learner_agent(args=None, n=3, act_spaces=None, obs_spaces=None):
    logger.info("act_spaces:" + str(act_spaces))
    logger.info("obs_spaces:" + str(obs_spaces))
    agent = Agent(args, n, act_spaces, obs_spaces)
    return agent


def serve(agent):
    logger.info("serve")
    c = zmq.Context()
    s = c.socket(zmq.REP)
    s.bind('tcp://127.0.0.1:%d' % agent.args.port)
    logger.info("zmq bind at tcp://127.0.0.1:%d" % agent.args.port)
    i, iter, episode, stop_client_num, record_i = 0, 0, 0, 0, 0
    start = time.time()
    explore_start = time.time()
    episode_rews = [0] * agent.args.save_rate
    with agent.writer.as_default():
        while True:
            z = s.recv_pyobj()
            p = zlib.decompress(z)
            data = pickle.loads(p)
            [obs, action, next_obs, rew, done, terminal] = data
            agent.buffer.add(obs, action, rew, next_obs, done)
            if all(done) or terminal:
                episode += 1
                episode_rews[episode % agent.args.save_rate] = np.mean(rew)
                if episode % agent.args.save_rate == 0:
                    record_i += 1
                    tf.summary.scalar(
                        '1.performance/2.episode_avg_rew',
                        np.mean(episode_rews), record_i)
                    logger.info(
                        "[%5.2f%%]episode %-8d mean rew:%8.3f, %10.2fsecs" % (
                            episode * 100. / agent.args.num_episodes, episode,
                            np.mean(episode_rews), time.time() - start))
            if i % agent.args.batch_size == 0 and episode > agent.args.warm_up:
                iter += 1
                explore_time = time.time() - explore_start
                logger.debug(
                    "serve collect %d explore samples spend %.3f secs" % (
                        agent.args.batch_size, explore_time))
                tf.summary.scalar('3.time/2.explore', explore_time, iter)
                agent.learn(iter)
                end = time.time()
                explore_start = end
            action = agent.action(obs)
            p = pickle.dumps(action)
            i += 1
            if episode >= agent.args.num_episodes:
                stop_client_num += 1
                logger.info("i=%d, episode=%d" % (i, episode))
                s.send_pyobj(pickle.dumps("stop"))
                if stop_client_num >= agent.args.num_env:
                    agent.writer.close()
                    break
            else:
                s.send_pyobj(p)
    s.close()
