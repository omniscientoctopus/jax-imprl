import itertools
from functools import partial

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces


class SingleAgentWrapper:

    def __init__(self, env):
        self.env = env

        # Joint action space
        _iterables = [range(k) for k in env.per_agent_actions]
        action_space = list(itertools.product(*_iterables))
        self.joint_action_space = jnp.array([list(action) for action in action_space])
        self.num_joint_actions = len(self.joint_action_space)

        # temporary fix for obs_dim for MLP
        obs, _ = self.reset(jax.random.PRNGKey(0))
        self.obs_dim = obs.shape[0]

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):

        obs, state = self.env.reset(key)

        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, act):

        action = self.joint_action_space[act][0]

        obs, state, reward, terminated, truncated, info = self.env.step(
            key, state, action
        )

        return obs, state, reward, terminated, truncated, info

    def split_key(self, key):
        return self.env.split_key(key)

    def action_space(self):
        return spaces.Discrete(self.num_joint_actions)
