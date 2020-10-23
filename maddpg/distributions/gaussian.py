# -*- coding:utf-8 -*-

import numpy as np


class GaussianProcess:
    def __init__(self, mu=0., sigma=1., size=1, decay=0.95, decay_steps=10000):
        """
        sigma 标准差，每decay_steps递减
        """
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.decay = decay
        self.decay_steps = decay_steps
        self.n_steps = 0

    def sample(self):
        self.n_steps += 1
        if self.n_steps % self.decay_steps == 0:
            self.sigma = self.sigma * self.decay
        x = np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)
        return x


if __name__ == '__main__':
    p = GaussianProcess(0.1, size=4)
    states = []
    for i in range(1000):
        states.append(p.sample())
    print(states[0])
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
