import numpy as np


class Policy():
    def __init__(self, P):
        self.proba_matrix = P
        self.action_number = P.shape[1]
        self.state_number = P.shape[0]

    def sample_action(self, state):
        action_probs = self.proba_matrix[state]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action


class DeterministicPolicy(Policy):
    def __init__(self, L, action_number):
        self.action_map = L
        state_number = len(L)
        P = np.zeros((state_number, action_number))
        for i in range(state_number):
            P[i, L[i]] = 1
        super().__init__(P)

    def update_proba_matrix(self):
        self.proba_matrix = np.zeros((self.state_number, self.action_number))
        for i in range(self.state_number):
            self.proba_matrix[i, self.action_map[i]] = 1

    def getAction(self, s):
        return self.action_map[s]

    def setAction(self, s, a):
        self.action_map[s] = a
