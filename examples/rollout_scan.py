"""
This is an example of how to use `jax.lax.scan` to rollout an environment.

"""

import time

import chex
import jax
import jax.numpy as jnp

import jax_imprl.envs

# Environment
ENV_NAME = "k_out_of_n_infinite"
ENV_SETTING = "4-of-4_infinite"
env = jax_imprl.envs.make(ENV_NAME, ENV_SETTING, single_agent=False, eval_env=True)


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

    # environment step
    action = jnp.array([0, 0, 0, 0])
    key, step_key = jax.random.split(runner.key)
    next_obs, env_state, reward, terminated, truncated, info = env.step(
        step_key, runner.env_state, action
    )

    ep_done = jnp.logical_or(terminated, truncated)

    # compute metrics
    metrics = {
        "returns": info["returns"],  # env_state.episode_return is reset to 0 when done
        "dones": ep_done,
    }

    # if done: update episode counter
    episode = jax.lax.cond(
        ep_done,
        lambda x: x + 1,
        lambda x: x,
        runner.ep,
    )

    # update the runner
    runner = Runner(
        key=key,
        env_state=env_state,
        obs=next_obs,
        ep=episode,
    )

    return runner, metrics


key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

runner = init_runner(key, env)

num_episodes = 10_000  # should take ~1s

time0 = time.time()
num_timesteps = env.time_horizon * num_episodes
runner, metrics = jax.block_until_ready(
    jax.lax.scan(update_runner, runner, length=num_timesteps)
)
time1 = time.time()

print(f"Time taken: {time1 - time0:.2f} seconds")

evals = metrics["returns"][metrics["dones"]]
mean = jnp.mean(evals)

print(f"Mean: {mean}")
