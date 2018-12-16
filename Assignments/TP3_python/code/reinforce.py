import numpy as np
from tqdm import tqdm

from policies import create_policy
import utils


"""
    REINFORCE algorithm implementation. This implementation assumes that the policy follows a Gaussian model with a fixed gamma.
    It could be adapted to other policy types in a relatively easy manner.
"""


class REINFORCE:
    """
    Parameters:
        - env: Environment
        - stepper: Gradient update stepper
        - policy type: string, defaults to 'gaussian'
        - N: number of episodes by iteration
        - T: horizon, number of time steps in a trajectory
        - n_itr: number of policy parameters updates
        - policy_params: dictionary of policy parameters
    """
    def __init__(self, env, stepper, policy_type='gaussian', N=100, T=100,
                 n_itr=100, gamma=0.9, sigma=0.4, exploration_bonus=False, beta=1.):
        self.env = env
        self.stepper = stepper
        self.N = N
        self.T = T
        self.n_itr = n_itr
        self.gamma = gamma
        self.policy_type = policy_type
        self.policy_params = {'sigma': sigma}

        if policy_type != 'gaussian':
            raise ValueError('Policy of type {} cannot be created'.format(policy_type))

        self.theta, self.theta_history, self.average_returns, self.counts = None, None, None, None  # initializing in constructor as None
        self.discounts = np.array([self.gamma ** t for t in range(T)])
        self.exploration_bonus = exploration_bonus
        self.beta = beta
        self.bin_state = 10
        self.bin_action = 20

    def _update_rewards(self, paths):
        bins = {}
        for n in range(self.N):
            for t in range(self.T):
                state, action = paths[n]['states'][t], paths[n]['actions'][t]
                bin = utils.discretization_2d(state, action, self.bin_state, self.bin_action)[0]
                bins[(n, t)] = bin
                self.counts[bin] = self.counts.get(bin, 0) + 1
        for n in range(self.N):
            for t in range(self.T):
                paths[n]['rewards'][t] += self.beta / np.sqrt(self.counts[bins[(n, t)]])

    """
    Compute the optimal parameters for the parametrized policy
    Parameters:
        - estimate_performance: if True, estimate the performance using utils.estimate_performances
    """
    def compute_optimal_policy(self, estimate_performance=True):
        if estimate_performance:
            self.average_returns = []

        self.theta = 0.
        self.theta_history = []
        self.theta_history.append(self.theta)

        self.counts = {}

        for _ in tqdm(range(self.n_itr)):
            policy = create_policy(self.policy_type, {**self.policy_params, 'theta': self.theta})

            paths = utils.collect_episodes(self.env, policy=policy, horizon=self.T, n_episodes=self.N)

            if self.exploration_bonus:
               self._update_rewards(paths)

            # performances for iteration
            if estimate_performance:
                self.average_returns.append(utils.estimate_performance(paths=paths))

            self.theta += self.stepper.update(
                policy.compute_J_estimated_gradient(
                    paths,
                    self.discounts,
                    N=self.N,
                    T=self.T,
                )
            )
            self.theta_history.append(self.theta[0])
