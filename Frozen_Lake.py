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

# Unified run interface with is_training flag
def run(
    episodes,
    algorithm='q_learning',
    is_training=True,
    render=False,
    slippery=False,
    seed=22
):
    """
    episodes: 
      - for Q-Learning: number of training episodes (if is_training=True) 
                       or evaluation episodes (if is_training=False)
      - for VI:        number of evaluation episodes (if is_training=False); 
                       ignored if is_training=True
    algorithm: 'q_learning' or 'value_iteration'
    is_training: whether to train (True) or load+evaluate (False)
    render:      render during evaluation
    slippery:    whether the lake is slippery
    seed:        RNG seed
    """
    # --- setup ---
    env = gym.make(
        'FrozenLake-v1',
        map_name="4x4",
        is_slippery=slippery,
        render_mode="human" if render and not is_training else None
    )
    base_env = env.unwrapped
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    rng = np.random.default_rng(seed)

    prefix = 'slippery' if slippery else 'non_slippery'
    q_file = f"q_{prefix}.pkl"
    vi_file = f"V_{prefix}.pkl"

    # --- initialize or load models ---
    if algorithm == 'q_learning':
        if is_training:
            q = np.zeros((env.observation_space.n, env.action_space.n))
        else:
            with open(q_file, 'rb') as f:
                q = pickle.load(f)
    else:  # value_iteration
        if is_training:
            V, policy, iters = value_iteration(base_env)
        else:
            with open(vi_file, 'rb') as f:
                V, policy = pickle.load(f)

    # Q-learning hyperparams
    alpha = 0.9
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 1e-4

    rewards_per_episode = np.zeros(episodes)

    # --- train or evaluate ---
    if algorithm == 'q_learning':
        if is_training:
            for i in range(episodes):
                state, _ = env.reset()
                done = False
                while not done:
                    if rng.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = int(np.argmax(q[state]))
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    q[state, action] += alpha * (
                        reward + gamma * np.max(q[new_state]) - q[state, action]
                    )
                    state = new_state
                epsilon = max(epsilon - epsilon_decay, 0)
                if reward == 1:
                    rewards_per_episode[i] = 1

            # save and report convergence
            with open(q_file, 'wb') as f:
                pickle.dump(q, f)
            conv_ep = check_convergence(rewards_per_episode)
            print(f"Q-Learning converged at episode {conv_ep}")
        else:
            # evaluation
            for i in range(episodes):
                state, _ = env.reset()
                done = False
                while not done:
                    action = int(np.argmax(q[state]))
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                if reward == 1:
                    rewards_per_episode[i] = 1
            print(f"Q-Learning success rate: {rewards_per_episode.mean()*100:.2f}%")
    else:  # value_iteration
        if is_training:
            # save VI results
            with open(vi_file, 'wb') as f:
                pickle.dump((V, policy), f)
            print(f"Value Iteration converged in {iters} iterations")
            env.close()
            return
        else:
            # evaluation
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
            print(f"VI policy success rate: {success/episodes*100:.2f}%")

    env.close()

    # --- plotting ---
    cum_success = np.cumsum(rewards_per_episode)
    plt.figure()
    plt.plot(cum_success)
    plt.title(f"Cumulative Successes ({algorithm})")
    plt.xlabel('Episode')
    plt.ylabel('Total Successes')
    plt.savefig(f"cum_success_{algorithm}_{prefix}.png")
    plt.show()

    # optional VI heatmap if you evaluated VI
    if algorithm == 'value_iteration' and not is_training:
        plt.figure()
        grid = int(np.sqrt(len(V)))
        V_grid = V.reshape((grid, grid))
        plt.imshow(V_grid, interpolation='nearest')
        for i in range(grid):
            for j in range(grid):
                plt.text(j, i, f"{V_grid[i, j]:.2f}", ha='center', va='center')
        plt.title('State-Value Heatmap (V)')
        plt.colorbar()
        plt.savefig(f"V_heatmap_{prefix}.png")
        plt.show()


if __name__ == "__main__":
    # Q-Learning train
    # run(15000, algorithm='q_learning', is_training=True, slippery=False)
    # Q-Learning eval
    run(1,  algorithm='q_learning', is_training=False, slippery=False, render=True)

    # VI train (just compute & save)
    # run(None,  algorithm='value_iteration', is_training=True, slippery=True)
    # VI eval
    run(1,  algorithm='value_iteration', is_training=False, slippery=True, render=True)
