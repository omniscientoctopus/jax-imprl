from functools import partial
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces
from jax import vmap


@struct.dataclass
class EnvState:
    timestep: float = 0.0
    episode_return: float = 0.0


class MatrixGame:

    def __init__(
        self,
        env_config,
        baselines=None,
        eval_env=False,
    ):

        self.reward_to_cost = env_config["reward_to_cost"]

        self.payoff = jnp.array(env_config["payoff_matrix"], dtype=jnp.float32)
        self.time_horizon = env_config["ep_length"]

        self.n_agents = self.payoff.ndim
        self.per_agent_actions = self.payoff.shape

        self.obs = jnp.zeros(self.n_agents, dtype=jnp.float32)

        self.baselines = baselines

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self,
        keys: chex.PRNGKey,
        state: EnvState,
        action: jnp.array,
    ) -> Tuple[chex.Array, float, bool, dict, EnvState]:

        # rewards
        reward = self.payoff[tuple(action)]

        # returns
        returns = state.episode_return + reward

        timestep = state.timestep + 1

        truncated = False
        terminated = timestep >= self.time_horizon
        done = jnp.logical_or(terminated, truncated)

        # info
        info = {"returns": returns}

        next_state = EnvState(
            timestep=timestep,
            episode_return=returns * jnp.logical_not(done),
        )

        return self.obs, next_state, reward, terminated, truncated, info

    def step(
        self, key, state, action
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        # environment step
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            None, state, action
        )

        # reset env
        obs_re, state_re = self.reset(None)

        # Auto-reset environment based on termination/truncation
        done = jnp.logical_or(terminated, truncated)
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, state, reward, terminated, truncated, info

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        return self.obs, EnvState(timestep=0.0, episode_return=0.0)

    def action_space(self) -> spaces.Discrete:
        dict = {str(i): spaces.Discrete(3) for i in range(self.n_agents)}
        return spaces.Dict(dict)

    @property
    def name(self) -> str:
        return "matrix_game"
