from lib.linear_model import *


def discounted_return(memory, gamma):
    return sum([gamma ** k * memory[k][2] for k in range(len(memory))])


def reinforce(env, policy, model, n_episodes, gamma=0.9):
    for _ in range(n_episodes):
        state = env.reset()
        memory = []
        done = False
        while not done:
            action_probs = estimator_policy.predict(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward, next_state, done))
            if done:
                break
            state = next_state

        for i in range(len(memory)):
            state, action, reward, next_state, done = memory[i]
            total_return = discounted_return(memory[i:], gamma)
            baseline_value = model.predict(state)
            advantage = total_return - baseline_value
            model.update(state, total_return)
            policy.update(state, advantage, action)
    return model
