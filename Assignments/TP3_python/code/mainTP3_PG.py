import lqg1d
import matplotlib.pyplot as plt
import numpy as np

from reinforce import REINFORCE
from steppers import ConstantLearningRate, ConstantLearningRateDecay


#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')

policy_type = 'gaussian'

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 100
# Each trajectory will have at most T time steps
T = 200
# Number of policy parameters updates
n_itr = 100
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 1e-4
# Optimal value for theta for discount = 0.9
optimal_theta = -0.59
# number of runs
n_runs = 10

#####################################################


def run_reinforce(N=N, learning_rate=learning_rate, beta=None):
    # define the update rule (stepper)
    stepper = ConstantLearningRate(lr=learning_rate)  # e.g., constant, adam or anything you want

    # fill the following part of the code with
    #  - REINFORCE estimate i.e. gradient estimate
    #  - update of policy parameters using the steppers
    #  - average performance per iteration
    #  -    distance between optimal mean parameter and the one at it k
    if beta is None:
        exploration_bonus = False
    else:
        exploration_bonus = True

    reinforce = REINFORCE(env, stepper, policy_type=policy_type, gamma=discount, N=N, T=T, n_itr=n_itr,
                          exploration_bonus=exploration_bonus, beta=beta)
    reinforce.compute_optimal_policy(estimate_performance=True)

    mean_parameters = reinforce.theta_history
    avg_return = reinforce.average_returns

    return mean_parameters, avg_return


betas = [None, 1., 0.1, 0.01]
mean_parameters, avg_return, std_mean_parameters, std_avg_return = {}, {}, {}, {}

for beta in betas:
    all_mean_parameters = []
    all_avg_return = []
    for _ in range(n_runs):
        results = run_reinforce(N=N, learning_rate=learning_rate, beta=beta)
        all_mean_parameters.append(results[0])
        all_avg_return.append((results[1]))
    mean_parameters[beta], avg_return[beta] = np.array(all_mean_parameters).mean(axis=0), np.array(all_avg_return).mean(axis=0)
    std_mean_parameters[beta], std_avg_return[beta] = np.array(all_mean_parameters).std(axis=0), np.array(all_avg_return).std(axis=0)

# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance

plt.figure()
for beta in betas:
    # plt.errorbar(
    #     list(range(len(avg_return[lr]))), avg_return[lr], std_avg_return[lr],
    #     linestyle='None', marker='x', label='lr = {}'.format(lr)
    # )
    plt.plot(avg_return[beta], label='Beta = {}'.format(beta))

plt.xlabel('Episodes (REINFORCE)')
plt.ylabel('Average returns with standard deviations')
plt.legend()


# plot the distance mean parameter
# of iteration k
plt.figure()
for beta in betas:
    plt.plot([min(np.abs(theta - optimal_theta), 10.) for theta in mean_parameters[beta]], label='Beta = {}'.format(beta))
plt.xlabel("Episodes (REINFORCE)")
plt.ylabel(r"$|\theta - \theta^{*}|$")
plt.legend()

plt.show()

import pdb; pdb.set_trace()
