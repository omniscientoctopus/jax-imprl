import time
import yaml
import itertools
import multiprocessing as mp

import numpy as np
import jax
import jax.numpy as jnp

import jax_imprl.envs
from numpy_k_out_of_n import KOutOfN as numpy_k_out_of_n


ACTIONS = [0, 1, 2, 0, 1]


def do_nothing_policy(env, obs):
    return ACTIONS


def numpy_rollout(env, do_nothing_policy):
    obs = env.reset()
    total_reward = 0
    done = False
    t = 0

    while not done:
        action = do_nothing_policy(env, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward * (env.discount_factor**t)
        t += 1
    return total_reward


def parallel_numpy_rollout(env, policy, rollout_method, num_episodes, verbose=False):

    # use all cores
    cpu_count = mp.cpu_count()

    # create an iterable for the starmap
    iterable = zip(
        itertools.repeat(env, num_episodes), itertools.repeat(policy, num_episodes)
    )
    with mp.Pool(cpu_count) as pool:
        list_func_evaluations = pool.starmap(rollout_method, iterable)

    results = np.hstack(list_func_evaluations)

    return results


def round_list(lst, decimals=2):
    return [round(x, decimals) for x in lst]


if __name__ == "__main__":
    # experiments = [1, 10, 100]
    experiments = [1, 10, 100, 1_000, 10_000]

    store_returns_for = experiments[-1]

    ENV_NAME = "k_out_of_n"
    ENV_SETTING = "5-of-5"

    ############################# NUMPY ################################
    config_path = f"../jax_imprl/envs/structural_envs/env_configs/{ENV_SETTING}.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    numpy_env = numpy_k_out_of_n(config, seed=12345)

    numpy_timings = []
    numpy_returns = []

    for NUM_EPISODES in experiments:
        start = time.time()

        for _ in range(NUM_EPISODES):
            total_reward = numpy_rollout(numpy_env, do_nothing_policy)

            if NUM_EPISODES == store_returns_for:
                numpy_returns.append(total_reward)

        end = time.time()

        numpy_timings.append(end - start)

    #################### NUMPY (multiprocessing) #######################

    numpy_mp_timings = []
    numpy_mp_returns = []

    numpy_env = numpy_k_out_of_n(config, seed=12345)

    for NUM_EPISODES in experiments:
        start = time.time()

        results = parallel_numpy_rollout(
            numpy_env, do_nothing_policy, numpy_rollout, NUM_EPISODES
        )

        if NUM_EPISODES == store_returns_for:
            numpy_mp_returns = results

        end = time.time()

        numpy_mp_timings.append(end - start)

    ########################## JAX (for loop) ##########################
    jax_env = jax_imprl.envs.make(
        ENV_NAME,
        ENV_SETTING,
        single_agent=False,
    )

    action = jnp.array(ACTIONS)

    jax_for_loop_timings = []
    jax_for_loop_returns = []

    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        # reset
        key, subkey = jax.random.split(key)
        obs, state = jax_env.reset(subkey)

        # rollout
        for _ in range(NUM_EPISODES):

            terminated, truncated = False, False
            total_reward = 0
            t = 0

            while not terminated and not truncated:

                # generate keys for next timestep
                key, step_keys = jax.random.split(key)
                obs, state, reward, terminated, truncated, _ = jax_env.step(
                    step_keys, state, action
                )

                total_reward += reward * (jax_env.discount_factor**t)
                t += 1

            if NUM_EPISODES == store_returns_for:
                jax_for_loop_returns.append(total_reward)

        end_jax = time.time()

        jax_for_loop_timings.append(end_jax - start_jax)

    ############################ JAX (scan) ############################

    import chex
    from functools import partial

    @chex.dataclass(frozen=True)
    class Runner:
        key: chex.PRNGKey
        env_state: chex.Array
        obs: chex.Array
        ep: int = 0

    def init_runner(key, env):

        # Initialize the environment
        key, env_rng = jax.random.split(key, 2)
        init_obs, env_state = env.reset(env_rng)

        return Runner(key=key, env_state=env_state, obs=init_obs)

    def update_runner(runner, metrics):

        # 1. Select action
        # currently only supports a fixed action for all episodes
        action = jnp.array(ACTIONS)

        # 2. Environment step
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, terminated, truncated, info = jax_env.step(
            step_key, runner.env_state, action
        )

        # 3. Update metrics
        # get from info because env_state.return is reset to 0 when done
        metrics = {
            "returns": info["returns"],
            "dones": jnp.logical_or(terminated, truncated),
        }

        # 4. Update runner state
        runner = Runner(key=key, env_state=env_state, obs=next_obs)

        return runner, metrics

    def scanned_rollout(key, episodes):

        runner = init_runner(key, jax_env)

        timesteps = jax_env.time_horizon * episodes

        runner, metrics = jax.block_until_ready(
            jax.lax.scan(update_runner, runner, length=timesteps)
        )

        return runner, metrics

    jax_scan_timings = []
    jax_scan_returns = []

    for NUM_EPISODES in experiments:

        start_jax = time.time()

        key = jax.random.PRNGKey(12345)

        runner, metrics = scanned_rollout(subkey, NUM_EPISODES)

        end_jax = time.time()

        jax_scan_timings.append(end_jax - start_jax)

        if NUM_EPISODES == store_returns_for:
            evals = metrics["returns"] * metrics["dones"]
            jax_scan_returns = evals[jnp.nonzero(evals)]

    ########################## JAX VMAP + SCAN ################################
    jax_vmap_timings = []
    jax_vmap_returns = []
    for NUM_EPISODES in experiments:
        key, key_ = jax.random.split(key)
        keys = jax.random.split(key_, NUM_EPISODES)
        start_jax_ = time.time()
        runners, metrics = jax.block_until_ready(
            jax.vmap(jax.jit(scanned_rollout, static_argnums=(1)), in_axes=(0, None))(
                keys, 1
            )
        )

        end_jax_ = time.time()
        jax_vmap_timings.append(end_jax_ - start_jax_)

        if NUM_EPISODES == store_returns_for:
            evals = metrics["returns"] * metrics["dones"]
            jax_vmap_returns = evals[jnp.nonzero(evals)]

    main_end = time.time()
    print(f"Total time: {main_end - start:.1f} s")

    ########################## Print results ###########################
    print(f"NumPy (for loop): {round_list(numpy_timings)}")
    print(f"NumPy (multiprocessing): {round_list(numpy_mp_timings)}")
    print(f"Jax (for loop): {round_list(jax_for_loop_timings)}")
    print(f"Jax (scan): {round_list(jax_scan_timings)}")
    print(f"Jax (vmap + scan): {round_list(jax_vmap_timings)}")

    def compare(list1, list2):
        return [l1 / l2 for l1, l2 in zip(list1, list2)]

    speedup_for_loop = compare(numpy_timings, jax_for_loop_timings)
    speedup_scan = compare(numpy_timings, jax_scan_timings)
    speedup_vmap = compare(numpy_timings, jax_vmap_timings)
    print("")
    print("Speedups wrt NumPy (for loop):")
    print(f"Speedup (Jax): {round_list(speedup_for_loop)}")
    print(f"Speedup (Jax scan): {round_list(speedup_scan)}")
    print(f"Speedup (Jax vmap + scan): {round_list(speedup_vmap)}")

    speedup_for_loop = compare(numpy_mp_timings, jax_for_loop_timings)
    speedup_scan = compare(numpy_mp_timings, jax_scan_timings)
    speedup_vmap = compare(numpy_mp_timings, jax_vmap_timings)
    print("")
    print("Speedups wrt NumPy (multiprocessing):")
    print(f"Speedup (Jax): {round_list(speedup_for_loop)}")
    print(f"Speedup (Jax scan): {round_list(speedup_scan)}")
    print(f"Speedup (Jax vmap + scan): {round_list(speedup_vmap)}")

    # Mean returns
    print("")
    mean_numpy_returns = np.mean(numpy_returns)
    mean_numpy_mp_returns = np.mean(numpy_mp_returns)
    mean_jax_for_loop_returns = np.mean(jax_for_loop_returns)
    mean_jax_scan_returns = np.mean(jax_scan_returns).item()
    mean_jax_vmap_returns = np.mean(jax_vmap_returns).item()
    mean_list = [
        mean_numpy_returns,
        mean_numpy_mp_returns,
        mean_jax_for_loop_returns,
        mean_jax_scan_returns,
        mean_jax_vmap_returns,
    ]
    print(f"Mean returns: {round_list(mean_list)}")
    rel_error = lambda x, y: abs(x - y) * 100 / x
    rel_error_numpy_mp = rel_error(mean_numpy_returns, mean_numpy_mp_returns)
    rel_error_jax_for_loop = rel_error(mean_numpy_returns, mean_jax_for_loop_returns)
    rel_error_jax_scan = rel_error(mean_numpy_returns, mean_jax_scan_returns)
    rel_error_jax_vmap = rel_error(mean_numpy_returns, mean_jax_vmap_returns)
    _list = [
        rel_error_numpy_mp,
        rel_error_jax_for_loop,
        rel_error_jax_scan,
        rel_error_jax_vmap,
    ]
    print(f"Relative error mean returns wrt NumPy: {round_list(_list)}")

    ########################## Plot results ############################
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    # Timing
    ax[0].plot(experiments, numpy_timings, ".--", label="NumPy")
    ax[0].plot(
        experiments,
        numpy_mp_timings,
        ".--",
        label=f"NumPy (multiprocessing, {mp.cpu_count()} cores)",
    )
    ax[0].plot(experiments, jax_for_loop_timings, ".-", label="Jax (for loop)")
    ax[0].plot(experiments, jax_scan_timings, ".-", label="Jax (scan)")
    ax[0].plot(experiments, jax_vmap_timings, ".-", label="Jax (vmap + scan)")

    ax[0].set_yscale("log")
    ax[0].set_yticks(
        [1e-2, 1e-1, 1e0, 1e1, 1e2], ["0.01 s", "0.1 s", "1 s", "10 s", "100 s"]
    )
    ax[0].set_xscale("log")
    ax[0].set_xticks(experiments, [str(x) for x in experiments])
    ax[0].set_title("Time taken vs Number of episodes")
    ax[0].set_xlabel("Number of episodes")
    ax[0].set_ylabel("Time (s)")
    ax[0].legend()

    # Returns
    ax[1].hist(numpy_returns, label="NumPy")
    ax[1].hist(
        numpy_mp_returns,
        label=f"NumPy (multiprocessing, {mp.cpu_count()} cores)",
        fill=False,
    )
    ax[1].hist(
        jax_for_loop_returns,
        label="Jax (for loop)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:orange",
    )
    ax[1].hist(
        jax_scan_returns,
        label="Jax (scan)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:green",
    )
    ax[1].hist(
        jax_vmap_returns,
        label="Jax (vmap + scan)",
        alpha=0.5,
        fill=False,
        edgecolor="tab:red",
    )

    ax[1].set_xlabel("Return")
    ax[1].set_title(f"Returns for {store_returns_for:,} episodes")
    ax[1].legend()

    plt.show()
