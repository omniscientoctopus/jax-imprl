import time

import jax
import jax.numpy as jnp

import jax_imprl.envs

# Environment
ENV_NAME = "k_out_of_n_infinite"
ENV_SETTING = "4-of-4_infinite"
env = jax_imprl.envs.make(ENV_NAME, ENV_SETTING, single_agent=False, eval_env=True)

action = jnp.array([0, 0, 0, 0])

key = jax.random.PRNGKey(0)

# reset
key, subkey = jax.random.split(key)
obs, state = env.reset(subkey)

time0 = time.time()
store = []
for ep in range(10_000):

    terminated, truncated = False, False
    total_reward = 0
    t = 0

    while not terminated and not truncated:

        # step the environment
        key, step_key = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = env.step(
            step_key, state, action
        )

        total_reward += reward * env.discount_factor**t
        t += 1

    print(f"Episode: {ep}, Total Reward: {total_reward}")
    store.append(total_reward)

print(f"Time taken: {time.time() - time0}")

mean = jnp.mean(jnp.array(store))

print(f"Mean: {mean}")
