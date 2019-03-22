from lib.mp import *
import numpy as np


class MRP(MP):
    def __init__(self, P, R, gamma):
        MP.__init__(self, P)
        self.reward = R
        self.gamma = gamma
        self.state_number = self.transition_matrix.shape[0]

    def get_value_function(self):
        self.value = np.dot(np.linalg.inv(np.identity(self.state_number) - self.gamma * self.transition_matrix),
                            np.transpose(self.reward))
        return self.value


class MRP_2(MP):
    def __init__(self, P, R_m, gamma):
        MP.__init__(self, P)
        self.transition_reward = R_m
        self.gamma = gamma
        self.state_number = self.transition_matrix.shape[0]

    def get_reward_per_state(self):
        self.reward = []
        for s in range(self.state_number):
            self.reward.append(sum([np.array(self.transition_matrix)[s][s_p] * np.array(self.transition_reward)[s][s_p] \
                                    for s_p in range(self.state_number)]))
        return MRP(self.transition_matrix, self.reward, self.gamma)
