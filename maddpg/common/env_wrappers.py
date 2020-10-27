# -*- coding:utf-8 -*-
import numpy as np

from maddpg.common.env_utils import make_env


class BatchedEnvironment:
    """A wrapper that batches several environment instances."""

    def __init__(self, args, id_offset=0):
        """Initialize the wrapper.
        Args:
          create_env_fn: A function to create environment instances.
          batch_size: The number of environment instances to create.
          id_offset: The offset for environment ids. Environments receive
          sequential ids starting from this offset.
        """
        self._batch_size = args.env_batch_size
        # Note: some environments require an argument to be of a native Python
        # numeric type. If we create env_ids as a numpy array,its elements will
        # be of type np.int32. So we create it as a plain Python array first.
        env_ids = [id_offset + i for i in range(args.env_batch_size)]
        self._envs = [make_env(args, id) for id in env_ids]
        self._env_ids = np.array(env_ids, np.int32)
        self._obs = None

    @property
    def env_ids(self):
        return self._env_ids

    @property
    def envs(self):
        return self._envs

    def step(self, action_batch):
        """Does one step for all batched environments sequentially."""
        rewards = []
        dones = []
        infos = []
        for i in range(self._batch_size):
            # action = [act for act in action_batch[i]]
            obs, reward, done, info = self._envs[i].step(action_batch[i])
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self._obs[i] = self.merge(obs)
        return self._obs, rewards, dones, infos

    def reset(self):
        """Reset all environments."""
        observations = [env.reset() for env in self._envs]
        self._obs = observations
        return self._obs

    def reset_if_done(self, done, terminal, episode_step, episode):
        """Reset the environments for which 'done' is True.
        Args:
          done: An array that specifies which environments are 'done',
          meaning their episode is terminated.
        Returns:
          Observations for all environments.
        """
        for i in range(self._batch_size):
            if all(done[i]) or terminal[i]:
                episode[i] += 1
                episode_step[i] = 0
                self._obs[i] = self.envs[i].reset()
        return self._obs

    def render(self, mode='human', **kwargs):
        # Render only the first one
        self._envs[0].render(mode, **kwargs)

    def close(self):
        for env in self._envs:
            env.close()

    def uniform_action(self):
        acts = []
        for e in self._envs:
            # TODO: 不同类型的action space
            act = [np.random.uniform(size=s.n) for s in e.action_space]
            acts.append(act)

        return acts

    def merge(self, item):
        return np.concatenate(item)
