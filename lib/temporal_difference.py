from collections import defaultdict


def td_zero(env, policy, n=10, gamma=0.9, alpha=0.9):
    state_value = defaultdict(float)
    for _ in range(n):
        state = env.reset()
        done = False
        while not done:
            action = policy.sample_action(state)
            next_state, reward, done, _ = env.step(state, action)
            delta_t = reward + gamma * state_value[next_state] - state_value[state]
            state_value[state] += alpha * delta_t
            state = next_state
    return state_value


def forward_td_lambda(env, policy, n_episodes=10, lambd=0.9, gamma=0.9, alpha=0.9):
    def n_step_return(memory, n, gamma, end_val):
        return sum([gamma ** k * memory[k][1] for k in range(n)]) + gamma ** n * end_val

    state_value = defaultdict(float)
    for _ in range(n_episodes):
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
            s, _, _ = memory[i]
            gt_lambda = (1 - lambd) * sum(
                [lambd ** (k - 1) * n_step_return(memory[i:], k, gamma, state_value[memory[i + k][0]]) for k in
                 range(len(memory) - i - 1)])
            state_value[state] += alpha * (gt_lambda - state_value[state])
    return state_value


def backward_td_lambda(env, policy, n=10, gamma=0.9, T=100, alpha=0.9, lambd=0.9):
    state_value = defaultdict(float)
    for _ in range(n):
        state = env.reset()
        done = False
        eligibility_trace = defaultdict(float)
        while not done:
            action = policy.sample_action(state)
            next_state, reward, done, _ = env.step(state, action)
            for s in eligibility_trace.keys():
                eligibility_trace[s] = lambd * gamma * eligibility_trace[s] + float(int(state == s))
            delta_t = reward + gamma * state_value[next_state] - state_value[state]
            state_value[state] += alpha * eligibility_trace[state] * delta_t
            state = next_state
    return state_value
