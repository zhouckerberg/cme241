from collections import defaultdict


def monte_carlo(env, policy, n_episodes=5, gamma=0.9):
    """
    This is an every visit Monte Carlo
    """
    state_counter = defaultdict(int)
    state_value = defaultdict(float)
    for episode in range(n_episodes):
        memory = []
        states = set()
        state = env.reset()
        done = False
        while not done:
            action = policy.sample_action(state)
            state, reward, done, _ = env.step(state, action)
            states.add(state)
            memory.append((state, action, reward))

        for i in range(len(memory)):
            state, _, reward = memory[i]
            G = discounted_return(memory[i:], gamma)
            state_counter[state] += 1
            state_value[state] += (G - state_value[state]) / state_counter[state]
    return state_value
