import numpy as np


class Env():
    def __init__(self, mdp):
        self.mdp = mdp
        self.counter = 0
        self.max_iter = 1000

    def reset(self):
        # Return the initial state
        self.counter = 0
        s = np.random.choice(np.arange(self.mdp.state_number))
        return s

    def step(self, state, action):
        # For a given action, step into next state
        mrp = self.mdp.mrp_list[action]
        state_probs = mrp.transition_matrix[state, :]
        next_state = np.random.choice(np.arange(len(state_probs)), p=state_probs)
        reward = mrp.reward[next_state]
        self.counter += 1
        done = self.counter > self.max_iter
        info = None
        return next_state, reward, done, info
