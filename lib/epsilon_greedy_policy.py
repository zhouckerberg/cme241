from lib.linear_model import *
import random


class EpsilonPolicy():
    def __init__(self, action_number, step_size=1e-2):
        self.approximator = LinearModel(action=True)
        self.epsilon = 1
        self.step_size = step_size
        self.action_number = action_number

    def sample_action(self, state):
        """
        We choose an action greedily.
        With proba epsilon we take a random action
        Otherwise we take the action with highest extimated Q
        """
        self.epsilon = max(0, self.epsilon - self.step_size)
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_number - 1)
        else:
            val_actions = []
            for a in range(self.action_number):
                val_a = self.approximator.predict(state, a)
                val_actions.append(val_a)
            return np.argmax(val_actions)

    def update(self, state, target, action):
        """
        Update the approximator function
        """
        self.approximator.update(state, target, action)
