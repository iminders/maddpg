import random

import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_n, action_n, reward_n, obs_next_n, done_n):
        data = (obs_n, action_n, reward_n, obs_next_n, done_n)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obs_n, actions, rewards, obs_next_n, dones = [], [], [], [], []
        for i in idxes:
            obs, action, reward, obs_next, done = self._storage[i]
            obs_n.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obs_next_n.append(np.array(obs_next, copy=False))
            dones.append(done)
        return np.array(obs_n), np.array(actions), np.array(rewards), \
            np.array(obs_next_n), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(
            0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(
            batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def get_train_input_data(self, batch_size):
        replay_sample_index = self.make_index(batch_size)
        # collect replay sample from all agents
        return self.sample_index(replay_sample_index)


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        # for all priority values
        self.capacity = capacity
        """
        [----Parent nodes----][----leaves to recode priority----]
        size: capacity - 1      size: capacity
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
           // \\
          1     2
        // \\ // \\
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        self.tree_node_num = 2 * capacity - 1
        self.tree = np.zeros(self.tree_node_num)
        # [----data frame----]
        #    size: capacity
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        # update data_frame
        self.data[self.data_pointer] = data
        # update tree_frame
        self.update(tree_idx, p)

        self.data_pointer += 1
        # replace when exceed the capacity
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
           // \\
          1     2
        // \\ // \\
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        # the while loop is faster than the method in the reference code
        while True:
            # this leaf's left and right kids
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # reach bottom, end search
            if cl_idx >= self.tree_node_num:
                leaf_idx = parent_idx
                break
            # downward search, always search for a higher priority node
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PrioritizedReplayBuffer(object):
    """
    stored as ( s, a, r, s_ ) in SumTree
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    # small amount to avoid zero priority
    epsilon = 0.01
    # [0~1] convert the importance of TD error to priority
    alpha = 0.6
    # importance-sampling, from initial value increasing to 1
    beta = 0.4
    beta_increment_per_sampling = 0.001
    # clipped abs error
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(int(capacity))

    def add(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        # set the max p for new p
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        obss, acts, rews, next_obss, dones = [], [], [], [], []
        # priority segment
        pri_seg = self.tree.total_p / n
        # max = 1
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # for later calculate ISweight
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            obs, act, rew, next_obs, done = data
            obss.append(obs)
            acts.append(act)
            rews.append(rew)
            next_obss.append(next_obs)
            dones.append(done)

        return np.array(obss), np.array(acts), np.array(rews), \
            np.array(next_obss), np.array(dones), b_idx, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
