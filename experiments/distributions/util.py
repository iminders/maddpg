from maddpg.distributions.gaussian import GaussianProcess
from maddpg.distributions.ou import OrnsteinUhlenbeckProcess


def get_distribution(act_space, noise_distribution="gaussian"):
    """ probability distribution """
    from gym import spaces
    if isinstance(act_space, spaces.Box):
        assert len(act_space.shape) == 1
        if noise_distribution == "gaussian":
            return GaussianProcess(size=act_space.shape[0])
        elif noise_distribution == "ou":
            return OrnsteinUhlenbeckProcess(size=act_space.shape[0])
    else:
        raise NotImplementedError
