import time
import yaml

import jax
import jax.numpy as jnp

import jax_imprl.structural_envs
from numpy_k_out_of_n import KOutOfN as numpy_k_out_of_n


def do_nothing_policy(env, obs):
    return [0, 0, 0, 0, 0]


def numpy_rollout(env, do_nothing_policy):
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = do_nothing_policy(env, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


def round_list(lst, decimals=2):
    return [round(x, decimals) for x in lst]


if __name__ == "__main__":
    # experiments = [1, 10, 100]
    experiments = [1, 10, 100, 1_000, 10_000]

    store_returns_for = experiments[-1]

    # NUMPY
    setting = "hard-5-of-5"
    config_path = f"../jax_imprl/structural_envs/env_configs/{setting}.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    env = numpy_k_out_of_n(config)

    numpy_timings = []
    numpy_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = numpy_rollout(env, do_nothing_policy)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    # JAX
    jax_env = jax_imprl.structural_envs.make(setting=setting)

    action = jnp.array([0, 0, 0, 0, 0])

    jax_timings = []
    jax_returns = []

    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        # rollout
        for _ in range(NUM_EPISODES):

            # reset
            key, subkey = jax.random.split(key)
            obs, state = jax_env.reset(subkey)

            done = False
            total_reward = 0

            while not done:

                # generate keys for next timestep
                key, step_keys = jax_env.split_key(key)
                obs, state, reward, done, _ = jax_env.step_env(step_keys, state, action)

                total_reward += reward

            if NUM_EPISODES == store_returns_for:
                jax_returns.append(total_reward)

        end_jax = time.time()

        jax_timings.append(end_jax - start_jax)

    # Print results
    print(f"NumPy timings: {round_list(numpy_timings)}")
    print(f"Jax timings: {round_list(jax_timings)}")
    speedup = [numpy / jax for numpy, jax in zip(numpy_timings, jax_timings)]
    print(f"Speedup: {round_list(speedup)}")

    # Plot results
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    # Timing
    ax[0].plot(experiments, numpy_timings, ".-", label="NumPy")
    ax[0].plot(experiments, jax_timings, ".-", label="Jax")

    ax[0].set_xlabel("Number of episodes")
    ax[0].set_ylabel("Time (s)")
    ax[0].legend()

    # Returns
    ax[1].hist(numpy_returns, label="NumPy")
    ax[1].hist(jax_returns, label="Jax", alpha=0.5, fill=False, edgecolor="tab:orange")

    ax[1].set_xlabel("Return")
    ax[1].set_title(f"Returns for {store_returns_for} episodes")
    ax[1].legend()

    plt.show()
