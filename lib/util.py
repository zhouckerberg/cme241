import numpy as np


def generate_stochastic_matrix(n):
    P = np.random.rand(n, n)
    return P / P.sum(axis=1)[:, None]


def generate_reward_matrix(n):
    return np.random.rand(n, n)


def generate_reward_vector(n):
    return np.random.rand(n)


def discounted_return(memory, gamma):
    return sum([gamma ** k * memory[k][1] for k in range(len(memory))])
