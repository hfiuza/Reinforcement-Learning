import numpy as np
import arms
from SMAB_algorithms import UCB1, TS, naive_sampling
import matplotlib.pyplot as plt

# Build your own bandit problem

# this is an example, please change the parameters or arms!


def construct_Bernoulli_MAB(difficulty='moderate'):
    if difficulty == 'moderate':
        arm1 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
        arm2 = arms.ArmBernoulli(0.25, random_state=np.random.randint(1, 312414))
        arm3 = arms.ArmBernoulli(0.20, random_state=np.random.randint(1, 312414))
        arm4 = arms.ArmBernoulli(0.10, random_state=np.random.randint(1, 312414))
    elif difficulty == 'hard':
        arm1 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
        arm2 = arms.ArmBernoulli(0.29, random_state=np.random.randint(1, 312414))
        arm3 = arms.ArmBernoulli(0.29, random_state=np.random.randint(1, 312414))
        arm4 = arms.ArmBernoulli(0.29, random_state=np.random.randint(1, 312414))
    elif difficulty == 'easy':
        arm1 = arms.ArmBernoulli(0.90, random_state=np.random.randint(1, 312414))
        arm2 = arms.ArmBernoulli(0.15, random_state=np.random.randint(1, 312414))
        arm3 = arms.ArmBernoulli(0.10, random_state=np.random.randint(1, 312414))
        arm4 = arms.ArmBernoulli(0.05, random_state=np.random.randint(1, 312414))
    else:
        raise ValueError('Difficulty {} is not supported'.format(difficulty))
    return [arm1, arm2, arm3, arm4]


def construct_non_parametric_MAB():
    arm1 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
    arm2 = arms.ArmBeta(0.5, 0.5, random_state=np.random.randint(1, 312414))
    arm3 = arms.ArmBeta(1., 3., random_state=np.random.randint(1, 312414))
    arm4 = arms.ArmExp(1., random_state=np.random.randint(1, 312414))
    arm5 = arms.ArmFinite(np.array([0., 0.1, 0.5, 0.8]), np.array([0.2, 0.3, 0.4, 0.1]))
    return [arm1, arm2, arm3, arm4, arm5]


MAB = construct_non_parametric_MAB()

# bandit : set of arms

nb_arms = len(MAB)
means = [el.mean for el in MAB]

# Display the means of your bandit (to find the best)
print('means: {}'.format(means))
mu_max = np.max(means)

# (Expected) regret curve for UCB, Thompson Sampling, and naive sampling

T = 5000  # horizon

reg1s, reg2s, reg3s = [], [], []

for idx in range(200):
    if idx % 10 == 0:
        print(idx)
    rew1, draws1 = UCB1(T, MAB)
    reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
    rew2, draws2 = TS(T, MAB)
    reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)

    rew3, draws3 = naive_sampling(T, MAB)
    reg3 = mu_max * np.arange(1, T + 1) - np.cumsum(rew3)

    reg1s.append(reg1)
    reg2s.append(reg2)
    reg3s.append(reg3)

# add oracle t -> C(p)log(t)


def kullback_leibler_divergence(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def problem_complexity(MAB):
    complexity = sum([(mu_max - arm.mean) / kullback_leibler_divergence(mu_max, arm.mean) for arm in MAB if mu_max != arm.mean])
    print('This problem has complexity {}'.format(complexity))
    return complexity


oracle = problem_complexity(MAB) * np.array([np.log(t) for t in np.arange(1, T + 1)])

plt.figure(1)
x = np.arange(1, T+1)
plt.plot(x, np.array(reg1s).mean(axis=0), label='UCB')
plt.plot(x, np.array(reg2s).mean(axis=0), label='Thompson')
plt.plot(x, np.array(reg3s).mean(axis=0), label='Naive')
plt.plot(x, oracle, label='Oracle')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.savefig('img/')
plt.show()
