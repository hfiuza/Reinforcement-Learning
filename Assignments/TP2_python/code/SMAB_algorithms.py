import numpy as np


def UCB1_strategy(t, Ss, Ns, ro=0.2):
    return np.argmax([s/n + ro * np.sqrt(np.log(t) / (2 * n)) for s, n in zip(Ss, Ns)])


def TS_strategy(Ss, Ns):
    return np.argmax([np.random.beta(s + 1, n - s + 1, 1) for n, s in zip(Ns, Ss)])


def naive_strategy(Ss, Ns):
    return np.argmax([s/n for s, n in zip(Ss, Ns)])


def bernoulli_sample(p):
    return np.random.binomial(size=1, n=1, p=p)[0]


def simulate(MAB, T, select_arm_strategy=None, integer_cumulative_rewards=False):
    K = len(MAB)
    # initialization
    Ns = [0] * K
    Ss = [0.] * K
    rewards, draws = [], []

    for i, arm in enumerate(MAB):
        rew = float(arm.sample())
        # save results
        rewards.append(rew)
        draws.append(i)
        # update distributions
        Ns[i] += 1
        Ss[i] = rew if not integer_cumulative_rewards else bernoulli_sample(rew)

    # Simulation
    for t in range(K, T):
        idx = select_arm_strategy(t, Ss, Ns)
        rew = float(MAB[idx].sample())
        # save results
        rewards.append(rew)
        draws.append(idx)
        # update distributions
        Ns[idx] += 1
        s_increment = rew if not integer_cumulative_rewards else bernoulli_sample(rew)
        Ss[idx] += s_increment

    return rewards, draws


def UCB1(T, MAB, ro=0.2):
    return simulate(MAB, T, select_arm_strategy=lambda t, Ss, Ns: UCB1_strategy(t, Ss, Ns, ro=ro))


def TS(T, MAB):
    return simulate(MAB, T, select_arm_strategy=lambda t, Ss, Ns: TS_strategy(Ss, Ns), integer_cumulative_rewards=True)


def naive_sampling(T, MAB):
    return simulate(MAB, T, select_arm_strategy=lambda t, Ss, Ns: naive_strategy(Ss, Ns))