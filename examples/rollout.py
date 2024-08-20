import time

import jax
import jax.numpy as jnp

import jax_imprl.structural_envs as structural_envs


# Environment
env = structural_envs.make(setting="hard-5-of-5")

key = jax.random.PRNGKey(42)

action = jnp.array([0, 0, 0, 0, 0])

time0 = time.time()
store = []
for ep in range(100):

    # reset
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    done = False
    total_reward = 0

    while not done:

        # generate keys for next timestep
        key, step_keys = env.split_key(key)
        obs, state, reward, done, _ = env.step_env(step_keys, state, action)

        total_reward += reward

    print(f"Episode: {ep}, Total Reward: {total_reward}")
    store.append(total_reward)

print(f"Time taken: {time.time() - time0}")
