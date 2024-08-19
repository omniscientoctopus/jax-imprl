import itertools

import numpy as np
import jax
import chex
from functools import partial
import jax.numpy as jnp

from gymnax.environments import spaces


class JointWrapper:

    def __init__(self, env):
        self.env = env

        action_space = list(
            itertools.product(np.arange(env.n_comp_actions), repeat=env.n_components)
        )
        self.joint_action_space = jnp.array([list(action) for action in action_space])
        self.num_joint_actions = len(self.joint_action_space)

        # temporary fix for obs_dim for MLP
        obs, _ = self.reset(jax.random.PRNGKey(0))
        self.obs_dim = obs.shape[0]

    def reset(self, key: chex.PRNGKey):

        obs, state = self.env.reset(key)

        return self.flatten(obs), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, act):

        action = self.joint_action_space[act][0]

        obs, state, reward, done, info = self.env.step(key, state, action)

        obs = self.flatten(obs)

        return obs, state, reward, done, info

    def flatten(self, obs):
        return jnp.concatenate([obs[0], obs[1].flatten()])

    def split_key(self, key):
        return self.env.split_key(key)

    def action_space(self):
        return spaces.Discrete(self.num_joint_actions)
