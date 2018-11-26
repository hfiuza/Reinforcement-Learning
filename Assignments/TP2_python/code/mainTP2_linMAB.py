import numpy as np
from linMAB_algorithms import RandomStrategy, EpsilonGreedyStrategy, LinUCBStrategy
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm

random_state = np.random.randint(0, 24532523)
model = ToyLinearModel(
    n_features=8,
    n_actions=20,
    random_state=random_state,
    noise=0.1)

# model = ColdStartMovieLensModel(
#     random_state=random_state,
#     noise=0.1
# )

n_a = model.n_actions
d = model.n_features

T = 6000


nb_simu = 20  # you may want to change this!

##################################################################
# define the algorithms
# - Random
# - Linear UCB
# - Eps Greedy
# and test it!
##################################################################

alg_name_to_strategy = {
    'random': (RandomStrategy, [{}]),
    'e-greedy': (EpsilonGreedyStrategy, [{'epsilon': value} for value in [0.01]]),
    'linUCB': (LinUCBStrategy, [{'alpha': value, 'dynamic_alpha': dynamic_alpha} for value in [0.3] for dynamic_alpha in [True]])
}

alg_names = ['random', 'e-greedy', 'linUCB']

mean_norms_by_alg_name = {}
mean_regret_by_alg_name = {}

print('real theta: {}'.format(model.real_theta))

for alg_name in alg_names:
    strategy_class, kwargs_list = alg_name_to_strategy[alg_name]

    for kwargs in kwargs_list:
        regret = np.zeros((nb_simu, T))
        norm_dist = np.zeros((nb_simu, T))

        for k in tqdm(range(nb_simu), desc="Simulating {}".format(alg_name)):

            strategy = strategy_class(model, **kwargs)

            for t in range(T):
                a_t = strategy.select()  # algorithm picks the action
                r_t = model.reward(a_t)  # get the reward

                # do something (update algorithm)
                strategy.update(a_t, r_t)
                theta_hat = strategy.estimate_theta()

                # store regret
                regret[k, t] = model.best_arm_reward() - r_t
                norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

        # compute average (over sim) of the algorithm performance and plot it
        mean_norms = norm_dist.mean(axis=0)
        mean_regret = regret.mean(axis=0)

        # append to dictionary
        mean_norms_by_alg_name[alg_name + str(kwargs)] = mean_norms
        mean_regret_by_alg_name[alg_name + str(kwargs)] = mean_regret

plt.figure(1)
plt.subplot(121)
for alg_name in mean_norms_by_alg_name:
    plt.plot(mean_norms_by_alg_name[alg_name], label=alg_name.split('{')[0])
plt.ylabel('d(theta, theta_hat)')
plt.xlabel('Rounds')
plt.legend()

plt.subplot(122)
for alg_name in mean_regret_by_alg_name:
    plt.plot(mean_regret_by_alg_name[alg_name].cumsum(), label=alg_name.split('{')[0])
plt.yscale('log')
plt.xlabel('Rounds')
plt.legend()
plt.show()
