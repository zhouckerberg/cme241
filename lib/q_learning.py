from collections import defaultdict
import numpy as np


def qLearning(env, n_episodes, policy, gamma, alpha=0.1):
    Q = defaultdict(float)
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action_proba = policy.proba_matrix[state]
            action = np.random.choice(np.arange(len(action_proba)), p=action_proba)
            next_state, reward, done, _ = env.step(state, action)
            next_action_proba = policy.proba_matrix[next_state]
            next_action = np.random.choice(np.arange(len(next_action_proba)), p=next_action_proba)
            Q[(state, action)] += alpha * (
                    reward + gamma * max([Q[(next_state, a)] for a in range(len(action_proba))]) - Q[
                (state, action)])
            action, state = next_action, next_state
    return Q
