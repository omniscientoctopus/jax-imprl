"""

Note: belief shape is (n_components, n_damage_states) unlike the numpy version where it is (n_damage_states, n_components) primarily because of the vmap implementation in JAX.

"""

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
    # Properties of components
    damage_state: jnp.array
    observation: jnp.array
    belief: jnp.array
    timestep: float
    returns: float


class JaxKOutOfN:
    """
    JAX implementation of the k-out-of-n system environment.

    Damage states:
    - 0: perfect
    - 1: minor damage
    - 2: major damage
    - 3: failure

    Actions:
    - 0: do nothing
    - 1: replace
    - 2: inspect

    """

    def __init__(self, env_config, wrapper="Filter", baselines=None):
        super().__init__()

        """
        Parameters
        ----------
        env_config: dict

        random_seed: int
                     random seed for reproducibility 

        """

        self.k = env_config["k"]
        self.wrapper = wrapper

        self.time_horizon = env_config["time_horizon"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]

        ######### component level ##########
        self.n_components = env_config["n_components"]  # number of components
        self.n_damage_states = env_config[
            "n_damage_states"
        ]  # damage states per component
        self.n_comp_actions = env_config["n_comp_actions"]  # actions per component

        # damage state transition probabilities
        # shape: (n_components, n_damage_states, n_damage_states)
        _array = np.array(env_config["transition_model"])
        # heterogeneous transition probabilities
        if _array.ndim == 3:
            self.deterioration_table = np.array(_array)
        # homogeneous transition probabilities
        elif _array.ndim == 2:
            self.deterioration_table = np.tile(_array, (self.n_components, 1, 1))

        # rewards of different actions for different components
        self.reward_model = np.zeros((self.n_components, self.n_comp_actions))
        # 3 available actions per component:
        # 0:do nothing, 1:replace, 2: inspect
        if env_config["identical"]:
            self.reward_model[:, 1] = [
                env_config["replacement_reward"]
            ] * self.n_components
            self.reward_model[:, 2] = [
                env_config["inspection_reward"]
            ] * self.n_components
        else:
            self.reward_model[:, 1] = env_config[
                "replacement_rewards"
            ]  #! Hyperparameter
            self.reward_model[:, 2] = env_config[
                "inspection_rewards"
            ]  #! Hyperparameter

        if env_config["identical"]:
            repair_accuracies = [env_config["replacement_accuracy"]] * self.n_components
        else:
            repair_accuracies = env_config["replacement_accuracies"]

        # repair transition model
        # describes transition for (imperfect) repairs
        repair_transition_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):

            r = repair_accuracies[c]
            repair_transition_model[c] = np.array(
                [[1, 0, 0, 0], [r, 1 - r, 0, 0], [r, 0, 1 - r, 0], [r, 0, 0, 1 - r]]
            )

        self.transition_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: __env_transition__
            # inspection: __env_transition__ + __inspect__
            self.transition_model[c, [0, 2]] = self.deterioration_table[c]

            # repair: __repair__ + __env_transition__
            self.transition_model[c, 1] = (
                repair_transition_model[c] @ self.deterioration_table[c]
            )

        # observation model
        # describes how accurately we perceive the underlying states
        # shape: [self.n_damage_states, self.n_observations]
        observation_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):

            if env_config["identical"]:
                p = env_config["obs_accuracy"]
            else:
                p = env_config["obs_accuracies"][c]

            observation_model[c] = np.array(
                [
                    [p, 1 - p, 0.0, 0.0],
                    [(1 - p) / 2, p, (1 - p) / 2, 0.0],
                    [0.0, 1 - p, p, 0.0],
                    [0.0, 0.0, 0.0, 1],
                ]
            )

        self.observation_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: __env_transition__
            self.failure_obs_model = jnp.array(
                [
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [1 / 3, 1 / 3, 1 / 3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # do nothing: only failure observation
            # replacement: only failure observation
            self.observation_model[c, [0, 1]] = self.failure_obs_model

            # repair: __repair__ + __env_transition__
            self.observation_model[c, 2] = observation_model[c]

        # convert to jax array
        self.reward_model = jnp.array(self.reward_model)
        self.transition_model = jnp.array(self.transition_model)
        self.observation_model = jnp.array(self.observation_model)

        self.component_list = jnp.arange(self.n_components)
        self.system_reward = self.reward_model[:, 1].sum()

        # baselines
        self.baselines = baselines

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _compute_transition(
        self,
        key_transition: chex.PRNGKey,
        component: int,
        dam_state: int,
        action: int,
    ) -> int:

        # In this case, we have no replacement
        next_dam_state = jax.random.choice(
            key_transition,
            self.n_damage_states,
            p=self.transition_model[component, action, dam_state],
        )

        return next_dam_state

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _compute_observation(
        self,
        key_obs: chex.PRNGKey,
        component: int,
        dam_state: int,
        action: int,
    ) -> int:

        obs = jax.random.choice(
            key_obs,
            self.n_damage_states,
            p=self.observation_model[component, action, dam_state],
        )

        return obs

    @partial(vmap, in_axes=(None, 0, 0))
    def _compute_action_rewards(self, component: int, action: int) -> float:
        return self.reward_model[component, action]

    def _compute_state_rewards(self, state: jnp.array) -> float:

        # count number of components in failed state
        num_failed = jnp.sum(state == self.n_damage_states - 1)
        num_operational = self.n_components - num_failed

        # k-out-of-n:G system is functional if at least k out of n
        # components are operational
        failure = jnp.less(num_operational, self.k)

        _penalty = self.system_reward * self.FAILURE_PENALTY_FACTOR * failure

        return _penalty

    def calculate_reward(self, state, action) -> float:
        action_rewards = self._compute_action_rewards(self.component_list, action).sum(
            axis=0
        )

        # System rewards
        state_rewards = self._compute_state_rewards(state.damage_state)

        reward = action_rewards + state_rewards
        return reward * self.discount_factor**state.timestep

    @partial(vmap, in_axes=(None, 0, 0, 0, 0))
    def _compute_belief_update(
        self,
        belief: jnp.array,
        component: int,
        obs: int,
        action: jnp.array,
    ) -> chex.Array:

        next_belief = self.transition_model[component, action, :, :].T @ belief
        state_probs = self.observation_model[component, action, :, obs]
        next_belief = next_belief * state_probs
        next_belief = next_belief / jnp.sum(next_belief)

        return next_belief

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self,
        keys: chex.PRNGKey,
        state: EnvState,
        action: jnp.array,
    ) -> Tuple[chex.Array, float, bool, dict, EnvState]:

        # split keys into keys for repair accuracies,
        # damage transitions and observations
        keys_transition, keys_obs = jnp.split(keys, 2, axis=0)

        # next state
        next_damage_state = self._compute_transition(
            keys_transition,
            self.component_list,
            state.damage_state,
            action,
        )

        # observation
        observation = self._compute_observation(
            keys_obs, self.component_list, next_damage_state, action
        )

        # belief update
        belief = self._compute_belief_update(
            state.belief, self.component_list, observation, action
        )

        # rewards
        reward = self.calculate_reward(state, action)
        returns = state.returns + reward

        timestep = state.timestep + 1

        # done
        done = self.is_terminal(timestep)

        # info
        info = {"returns": returns}

        next_state = EnvState(
            damage_state=next_damage_state,
            observation=observation,
            belief=belief,
            timestep=timestep,
            returns=returns * jnp.logical_not(done),
        )

        return self.get_obs(next_state), next_state, reward, done, info

    def step(
        self, key, state, action
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        key, subkeys = self.split_key(key)

        # environment step
        obs_st, state_st, reward, done, info = self.step_env(subkeys, state, action)

        # reset env
        key, sub_key = jax.random.split(key)
        obs_re, state_re = self.reset(sub_key)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )

        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:

        # initial damage state
        damage_state = jnp.zeros(self.n_components, dtype=jnp.int32)

        initial_belief = jnp.zeros(self.n_damage_states, dtype=jnp.float32)
        initial_belief = initial_belief.at[0].set(1.0)

        # initial belief
        belief = jnp.array([initial_belief] * self.n_components)

        env_state = EnvState(
            damage_state=damage_state,
            observation=damage_state,
            belief=belief,
            timestep=0.0,
            returns=0.0,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, state: EnvState) -> chex.Array:

        if self.wrapper == "OneHot":
            _one_hot = jnp.zeros(
                (self.n_components, self.n_damage_states), dtype=jnp.uint8
            )
            _one_hot = _one_hot.at[self.component_list, state.damage_state].set(1)
            obs = _one_hot

        elif self.wrapper == "Obs":
            obs = state.observation

        elif self.wrapper == "Filter":
            obs = state.belief

        return jnp.array([state.timestep / self.time_horizon]), obs

    def is_terminal(self, timestep: float) -> bool:
        return timestep >= self.time_horizon

    def action_space(self) -> spaces.Discrete:
        dict = {str(i): spaces.Discrete(3) for i in range(self.n_components)}
        return spaces.Dict(dict)

    def state_space(self):

        if self.wrapper == "Filter":
            dict = {
                str(i): spaces.Box(0, 1, (self.n_damage_states,), dtype=jnp.float32)
                for i in range(self.n_components)
            }
        elif self.wrapper == "OneHot" or self.wrapper == "Obs":
            dict = {
                str(i): spaces.Discrete(self.n_damage_states)
                for i in range(self.n_components)
            }

        dict["time"] = spaces.Box(0, 1, (1,), dtype=jnp.float32)
        return spaces.Dict(dict)

    @property
    def name(self) -> str:
        return "k-out-of-n"

    @partial(jax.jit, static_argnums=(0,))
    def split_key(self, key: chex.PRNGKey) -> Tuple[chex.PRNGKey, chex.PRNGKey]:
        """
        Split key into keys for each random variable:

        - keys for damage transitions of each component (#component)
        - keys for observations of each component (#component)
        - key for next timestep (1)

        """

        _num_RV = 2  # number of random variables

        keys = jax.random.split(key, self.n_components * _num_RV + 1)
        subkeys = keys[: self.n_components * _num_RV, :]
        key = keys[self.n_components, :]

        return key, subkeys
