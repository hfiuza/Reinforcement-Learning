import numpy as np
import random

class AbstractStrategy(object):
    def __init__(self, linMABmodel):
        self.linMABmodel = linMABmodel
        self.lambd = 0.1
        self.actions_history = []
        self.rewards_history = []
        self.estimated_theta= None

    def _update_theta(self):
        Z = np.array([self.linMABmodel.features[a] for a in self.actions_history])
        y = np.array(self.rewards_history)
        # import pdb; pdb.set_trace()
        self.inverse = np.linalg.inv(np.matmul(Z.T, Z) + self.lambd * np.eye(Z.shape[1]))
        self.estimated_theta = np.matmul(self.inverse, np.matmul(Z.T, y)).T[0]

    def update(self, a_t, r_t):
        self.actions_history.append(a_t)
        self.rewards_history.append(r_t)
        if len(self.actions_history) >= self.linMABmodel.n_features:
            self._update_theta()

    def select(self):
        pass

    def estimate_theta(self):
        return self.estimated_theta if self.estimated_theta is not None else np.zeros(self.linMABmodel.n_features)


class RandomStrategy(AbstractStrategy):
    def select(self):
        return random.choice(range(self.linMABmodel.n_actions))


class EpsilonGreedyStrategy(AbstractStrategy):
    def __init__(self, linMABmodel, epsilon=0.1):
        super(EpsilonGreedyStrategy, self).__init__(linMABmodel)
        self.epsilon = epsilon

    def select(self):
        if self.estimated_theta is None or random.random() < self.epsilon:
            return random.choice(range(self.linMABmodel.n_actions))

        estimates = [np.matmul(self.estimated_theta.T, self.linMABmodel.features[a]) for a in range(self.linMABmodel.n_actions)]
        return max(range(self.linMABmodel.n_actions), key=lambda a: estimates[a])


class LinUCBStrategy(AbstractStrategy):
    def __init__(self, linMABmodel, alpha=2., dynamic_alpha=True):
        super(LinUCBStrategy, self).__init__(linMABmodel)
        self.alpha = alpha
        self.dynamic_alpha = dynamic_alpha

    def select(self):
        if self.estimated_theta is None:
            return random.choice(range(self.linMABmodel.n_actions))
        estimates = [np.matmul(self.estimated_theta.T, self.linMABmodel.features[a]) for a in range(self.linMABmodel.n_actions)]
        alpha_t = self.alpha * np.sqrt(np.log(len(self.actions_history))) if self.dynamic_alpha else self.alpha
        betas = alpha_t * np.array([np.matmul(np.matmul(self.linMABmodel.features[a].T, self.inverse), self.linMABmodel.features[a]) for a in range(self.linMABmodel.n_actions)])
        upper_bound_estimates = [estimate + beta for estimate, beta in zip(estimates, betas)]
        return max(range(self.linMABmodel.n_actions), key=lambda a: upper_bound_estimates[a])
