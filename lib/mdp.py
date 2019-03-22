from lib.policy import *
from lib.mrp import *
from collections import defaultdict
import numpy as np


class MDP():
    def __init__(self, gamma, mrp_list, P_list=None, R_list=None):
        self.gamma = gamma
        self.mrp_list = mrp_list
        self.state_number = self.mrp_list[0].state_number
        self.action_number = len(self.mrp_list)

    def check_dim(self, action_number, state_number):
        return action_number == self.action_number and state_number == self.state_number

    def get_mrp(self, policy):
        if not self.check_dim(policy.action_number, policy.state_number):
            raise Exception('Policy dimensions and MDP dimensions do not match')
        P = np.zeros((self.state_number, self.state_number))
        R = np.zeros(self.state_number)
        for s in range(self.state_number):
            R[s] = sum([policy.proba_matrix[s, a] * self.mrp_list[a].reward[s] for a in range(self.action_number)])
            for s_p in range(self.state_number):
                P[s, s_p] = sum([policy.proba_matrix[s, a] * self.mrp_list[a].transition_matrix[s, s_p] for a in
                                 range(self.action_number)])
        return MRP(P, R, self.gamma)

    def policy_evaluation(self, policy, theta=1e-3):
        V = defaultdict(float)
        while True:
            delta = 0
            for s in range(self.state_number):
                v = 0
                for a, action_prob in enumerate(policy.proba_matrix[s]):
                    v += action_prob * self.mrp_list[a].reward[s]
                    for s_prime in range(self.state_number):
                        prob = self.mrp_list[a].transition_matrix[s, s_prime]
                        v += self.gamma * action_prob * prob * V[s_prime]
                delta = max(np.abs(v - V[s]), delta)
                V[s] = v
            if delta < theta:
                break
        return V

    def policy_iteration(self):
        policy = DeterministicPolicy([0 for _ in range(self.action_number)], self.state_number)
        V = defaultdict(float)
        improvement = True
        while improvement:
            improvement = False
            V = self.policy_evaluation(policy, self.gamma)
            print(policy.proba_matrix, V)
            for s in range(self.state_number):
                current_action = np.argmax(policy.proba_matrix[s])
                action_values = np.zeros(self.action_number)
                for a in range(self.action_number):
                    for s_prime in range(self.state_number):
                        prob = self.mrp_list[a].transition_matrix[s, s_prime]
                        action_values[a] = prob * (self.mrp_list[a].reward[s] + self.gamma * V[s_prime])
                best_action = np.argmax(action_values)

                if best_action != current_action:
                    improvement = True
                    print("Improvement", s, a, best_action)
                    policy.setAction(s, best_action)
                    policy.update_proba_matrix()
        return policy, V

    def value_iteration(self, epsilon=1e-1):
        delta = float('inf')
        V = np.zeros(self.state_number)
        while delta > epsilon:
            delta = 0
            for s in range(self.state_number):
                tmp = V[s]
                best_sum = -float('inf')
                for a in range(self.action_number):
                    candidate = sum(
                        [self.mrp_list[a].transition_matrix[s, s1] * (self.mrp_list[a].reward[s1] + self.gamma * V[s1])
                         for s1 in range(self.state_number)])
                    best_sum = max(best_sum, candidate)
                V[s] = best_sum
                delta = max(delta, np.abs(tmp - V[s]))
        return V


class MDP_2():
    def __init__(self, P_list, R_list, gamma):
        self.gamma = gamma
        self.mrp2_list = [MRP_2(P_list[i], R_list[i], gamma) for i in range(len(P_list))]

    def get_reward_per_state(self):
        return MDP(self.gamma, None, None, mrp_list=[mrp_2.get_reward_per_state() for mrp_2 in mrp2_list])
