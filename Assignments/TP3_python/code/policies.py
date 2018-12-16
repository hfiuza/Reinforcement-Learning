from abc import ABC, abstractmethod
import numpy as np


def create_policy(policy_type, policy_params):
    policy_from_name = {
        'gaussian': GaussianPolicy
    }
    if policy_type not in policy_from_name:
        raise ValueError('Policy of type {} cannot be created'.format(policy_type))
    return policy_from_name[policy_type](**policy_params)


class AbstractPolicy(ABC):

    @abstractmethod
    def draw_action(self, state):
        pass


class GaussianPolicy(AbstractPolicy):
    """
    Parameters:
        theta: mean
        sigma: standard deviation
    """
    def __init__(self, theta=0., sigma=0.4):
        self.theta = theta
        self.sigma = sigma

    def draw_action(self, state):
        mu = self.theta * state
        return np.random.normal(mu, self.sigma)

    def compute_J_estimated_gradient(self, paths, discounts, N, T):
        gradient = 0

        for n in range(N):
            for t in range(T):
                action, state = paths[n]["actions"][t], paths[n]["states"][t]

                log_policy_gradient = (action - self.theta * state) * state / (self.sigma ** 2)
                cumulative_reward = np.dot(discounts[t:], paths[n]["rewards"][t:])

                gradient += log_policy_gradient * cumulative_reward
        return gradient / N
