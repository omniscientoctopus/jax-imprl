"""
A runner module for the scanned episode rollout. 

Input: key, env, action, num_episodes
Output: episodic returns

Currently, the runner only supports a fixed action for all episodes. 
In the future, we will extend the runner to support a policy function 
that takes the observation as input and returns the action.

"""

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Runner:
    key: chex.PRNGKey
    env_state: chex.Array
    obs: chex.Array
    ep: int = 0


def scanned_rollout(key, env, action, num_episodes):
    # TODO: Implement agent policy

    action = jnp.array(action)

    def init_runner(key, env):

        # Initialize the environment
        key, env_rng = jax.random.split(key, 2)
        init_obs, env_state = env.reset(env_rng)

        return Runner(key=key, env_state=env_state, obs=init_obs)

    def update_runner(runner, metrics):

        # 1. Select action
        # currently only supports a fixed action for all episodes

        # 2. Environment step
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, terminated, truncated, info = env.step(
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

    runner = init_runner(key, env)

    num_timesteps = env.time_horizon * num_episodes
    runner, metrics = jax.block_until_ready(
        jax.lax.scan(update_runner, runner, length=num_timesteps)
    )

    return metrics["returns"][metrics["dones"]]
