import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


# Value Iteration





# Q-Learning

def check_convergence(rewards_per_episode, threshold=0.9, window_size=100):
    for i in range(window_size, len(rewards_per_episode)):
        window = rewards_per_episode[i - window_size:i]
        if np.mean(window) >= threshold:
            return i
    return None


def run(episodes, is_training=True, render=False, slippery=False, seed=22):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=slippery, render_mode="human" if render else None)
    
    q_file_name = f"frozen_lake8x8_{'slippery' if slippery else 'non_slippery'}.pkl"
    q_img_name = f"frozen_lake8x8_{'slippery' if slippery else 'non_slippery'}.png"

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    rng = np.random.default_rng(seed)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open(q_file_name, "rb")
        q = pickle.load(f)
        f.close()

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()
    success = 0

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] += learning_rate * (reward + discount_factor * np.max(q[new_state, :]) - q[state, action])

            state = new_state
        
        epsilon = max(epsilon - epsilon_decay, 0)

        if epsilon == 0:
            learning_rate = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1
            success += 1

    env.close()


    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):t+1])
    plt.plot(sum_rewards)
    plt.show()
    plt.savefig(q_img_name)

    if is_training:
        convergence_episode = check_convergence(rewards_per_episode)
        print(f"Converged at episode {convergence_episode}")

        f = open(q_file_name, "wb")
        pickle.dump(q, f)
        f.close()

    if not is_training:
        print(f"Success rate: {np.mean(rewards_per_episode) * 100}%")

if __name__ == "__main__":
    #run(15000, is_training=True, render=False, slippery=False)
    #run(1000, is_training=False, render=False, slippery=False)
    run(1, is_training=False, render=True, slippery=False)

    #run(15000, is_training=True, render=False, slippery=True)
    #run(1000, is_training=False, render=False, slippery=True)
    run(1, is_training=False, render=True, slippery=True)