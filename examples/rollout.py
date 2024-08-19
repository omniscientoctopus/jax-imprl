import time

import jax
import jax.numpy as jnp

import jax_imprl.structural_envs as structural_envs


# Environment
env = structural_envs.make(setting="hard-5-of-5")

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)

action = jnp.array([0, 0, 0, 0, 0])

time0 = time.time()
store = []
for ep in range(100):

    # Reset the environment
    obs, state = env.reset(subkey)
    key, subkey = jax.random.split(key)
    done = False
    total_reward = 0

    while not done:

        # step the environment
        obs, state, reward, done, info = env.step(subkey, state, action)

        total_reward += reward

        key, subkey = jax.random.split(key)

    print(f"Episode: {ep}, Total Reward: {total_reward}")
    store.append(total_reward)

print(f"Time taken: {time.time() - time0}")
