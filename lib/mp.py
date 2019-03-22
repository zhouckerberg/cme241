import numpy as np
from scipy.linalg import eig


class MP:
    def __init__(self, P):
        self.transition_matrix = P

    def stationary_distribution(self):
        S, U = eig(self.transition_matrix.T)
        stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stationary /= np.sum(stationary)
        return stationary.real
