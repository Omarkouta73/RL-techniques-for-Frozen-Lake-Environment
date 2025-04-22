import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Q-Learning Convergence Check
def check_convergence(rewards_per_episode, threshold=0.9, window_size=100):
    for i in range(window_size, len(rewards_per_episode)):
        if np.mean(rewards_per_episode[i - window_size:i]) >= threshold:
            return i
    return None

# Value Iteration Algorithm
def value_iteration(env, gamma=0.9, tol=1e-6):
    nS, nA = env.observation_space.n, env.action_space.n
    V = np.zeros(nS)
    iteration = 0
    while True:
        delta = 0
        for s in range(nS):
            # compute expected returns for all actions using base env.P
            action_values = [
                sum(p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a])
                for a in range(nA)
            ]
            v_new = max(action_values)
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        iteration += 1
        if delta < tol:
            break
    # extract greedy policy
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        q_sa = [
            sum(p * (r + gamma * V[s_next]) for p, s_next, r, _ in env.P[s][a])
            for a in range(nA)
        ]
        policy[s] = int(np.argmax(q_sa))
    return V, policy, iteration

# Unified run interface

def run(episodes, algorithm='q_learning', render=False, slippery=False, seed=22):
    """
    Unified interface for Q-Learning or Value Iteration on FrozenLake-v1.

    episodes: number of episodes to train (Q-learning) or to evaluate (VI).
    algorithm: 'q_learning' or 'value_iteration'.
    render: whether to render environment during evaluation.
    slippery: environment slipperiness.
    seed: random seed for reproducibility.
    """
    # Create environment
    env = gym.make(
        'FrozenLake-v1', map_name="4x4", is_slippery=slippery,
        render_mode="human" if render else None
    )
    # Unwrap to access transition model
    base_env = env.unwrapped

    # Set seeds
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    rng = np.random.default_rng(seed)

    # File names based on algorithm and slipperiness
    prefix = 'slippery' if slippery else 'non_slippery'

    # Initialize structures
    if algorithm == 'q_learning':
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        V, policy, iters = value_iteration(base_env)

    # Q-learning hyperparameters
    alpha = 0.9
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 1e-4

    rewards_per_episode = np.zeros(episodes)

    # Training or evaluation loop
    if algorithm == 'q_learning':
        for i in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                # epsilon-greedy action selection
                if rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = int(np.argmax(q[state]))
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Q-update
                q[state, action] += alpha * (
                    reward + gamma * np.max(q[new_state]) - q[state, action]
                )
                state = new_state
            epsilon = max(epsilon - epsilon_decay, 0)
            if reward == 1:
                rewards_per_episode[i] = 1
        # Save Q-table
        with open(f"q_{prefix}.pkl", "wb") as f:
            pickle.dump(q, f)
        # Convergence
        conv_ep = check_convergence(rewards_per_episode)
        print(f"Q-Learning converged at episode {conv_ep}")
    else:
        # Evaluate VI policy using original env for stepping
        success = 0
        for i in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = int(policy[state])
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if reward == 1:
                rewards_per_episode[i] = 1
                success += 1
        print(f"Value Iteration converged in {iters} iterations")
        print(f"VI policy success rate: {success/episodes*100:.2f}%")
        # Save policy and values
        with open(f"V_{prefix}.pkl", "wb") as f:
            pickle.dump((V, policy), f)

    env.close()

    # Cumulative success plot
    cum_success = np.cumsum(rewards_per_episode)
    plt.figure()
    plt.plot(cum_success)
    plt.title(f"Cumulative Successes ({algorithm})")
    plt.xlabel('Episode')
    plt.ylabel('Total Successes')
    plt.savefig(f"cum_success_{algorithm}_{prefix}.png")
    plt.show()

    # For VI, also plot heatmap of state values
    if algorithm == 'value_iteration':
        plt.figure()
        grid_shape = int(np.sqrt(len(V)))
        V_grid = V.reshape((grid_shape, grid_shape))
        plt.imshow(V_grid, interpolation='nearest')
        for i in range(grid_shape):  # annotate values
            for j in range(grid_shape):
                plt.text(j, i, f"{V_grid[i, j]:.2f}", ha='center', va='center')
        plt.title('State-Value Heatmap (V)')
        plt.colorbar()
        plt.savefig(f"V_heatmap_{prefix}.png")
        plt.show()


if __name__ == "__main__":
    # Example usages:
    # Q-Learning training:
    # run(15000, algorithm='q_learning', render=False, slippery=False)
    # Q-Learning evaluation:
    # run(1000, algorithm='q_learning', render=False, slippery=False)

    # Value Iteration evaluation:
    # run(10000, algorithm='value_iteration', render=False, slippery=False)
    run(30000, algorithm='value_iteration', render=False, slippery=True)
