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
    timestep: float = 0.0
    episode_return: float = 0.0


class JaxKOutOfN:
    """
    JAX implementation of the infinite-horizon k-out-of-n system environment.

    Damage states:
    - 0: perfect
    - 1: major damage
    - 2: failure

    Actions:
    - 0: do nothing
    - 1: replace
    - 2: inspect

    """

    def __init__(
        self,
        env_config,
        baselines=None,
        eval_env=False,
        wrapper="Filter",
        reward_shaping: bool = True,
    ):

        self.wrapper = wrapper
        self.reward_shaping = reward_shaping
        self.global_obs = False  # global observation

        # time limits for training and evaluation
        train_time_horizon = 50  # training time limit per episode
        test_time_horizon = 20  # evaluation time limit per episode
        self.time_normalise_factor = max(train_time_horizon, test_time_horizon) * 2
        self.time_horizon = test_time_horizon if eval_env else train_time_horizon

        self.k = env_config["k"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]

        ######### component level ##########
        self.n_components = env_config["n_components"]
        self.n_damage_states = env_config["n_damage_states"]
        self.n_comp_actions = env_config["n_comp_actions"]  # actions per component
        try:
            self.initial_belief = jnp.array(env_config["initial_belief"])
        except KeyError:
            print("Initial belief not specified")

        ######################### Reward Model #########################

        # rewards of different actions for different components
        self.reward_model = np.zeros((self.n_components, self.n_comp_actions))
        # 3 available actions per component:
        # 0:do nothing, 1:replace, 2: inspect
        self.reward_model[:, 1] = env_config["replacement_rewards"]
        self.reward_model[:, 2] = env_config["inspection_rewards"]

        self.system_replacement_reward = sum(env_config["replacement_rewards"])
        self.failure_cost = self.system_replacement_reward * self.FAILURE_PENALTY_FACTOR
        try:
            self.mobilisation_reward = env_config["mobilisation_reward"]
        except KeyError:
            print("Mobilisation reward not specified.")

        ####################### Transition Model #######################

        # shape: (n_components, n_damage_states, n_damage_states)
        self.deterioration_table = np.array(env_config["transition_model"])

        # replacement transition model
        # describes transition for (imperfect) replacements
        replacement_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            r = env_config["replacement_accuracies"][c]
            replacement_table[c] = np.array([[1, 0, 0], [r, 1 - r, 0], [r, 0, 1 - r]])

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

            # replacement: __replacement__ + __env_transition__
            self.transition_model[c, 1] = (
                replacement_table[c] @ self.deterioration_table[c]
            )

        ####################### Observation Model ######################

        # describes how accurately we perceive the underlying states
        # shape: [self.n_damage_states, self.n_observations]
        inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):

            p = env_config["obs_accuracies"][c]
            try:
                f_p = env_config["failure_obs_accuracies"][c]
            except KeyError:
                print("Failure observation accuracy not specified.")

            inspection_model[c] = np.array(
                [
                    [p, 1 - p, 0.0],
                    [(1 - p) / 2, p, (1 - p) / 2],
                    [0.0, 1 - f_p, f_p],
                ]
            )

        no_inspection_model = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3],
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
            # do nothing: only failure observation
            # replacement: only failure observation
            self.observation_model[c, [0, 1]] = no_inspection_model

            # replacement: __replacement__ + __env_transition__
            self.observation_model[c, 2] = inspection_model[c]

        # convert to jax array
        self.reward_model = jnp.array(self.reward_model)
        self.transition_model = jnp.array(self.transition_model)
        self.observation_model = jnp.array(self.observation_model)

        self.component_list = jnp.arange(self.n_components)

        # baselines
        self.baselines = baselines

    @staticmethod
    def pf_sys(pf, k):
        """Computes the system failure probability pf_sys for k-out-of-n components

        Args:
            pf: Numpy array with components' failure probability.
            k: Integer indicating k (out of n) components.

        Returns:
            PF_sys: Numpy array with the system failure probability.
        """

        n = pf.size
        nk = n - k
        m = k + 1
        A = jnp.zeros(m + 1, dtype=jnp.float32)
        A = A.at[1].set(1)
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                # A[m] = A[m] + A[k] * Rel
                A = A.at[m].set(A[m] + A[k] * Rel)
                h = k
            for i in range(h, L - 1, -1):
                # A[i] = A[i] + (A[i - 1] - A[i]) * Rel
                A = A.at[i].set(A[i] + (A[i - 1] - A[i]) * Rel)
        PF_sys = 1 - A[m]
        return PF_sys

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

    def _compute_state_rewards(self, state: jnp.array, belief: jnp.array) -> float:

        if self.reward_shaping:

            pf = belief[:, -1]
            pf_sys = self.pf_sys(pf, self.k)
            factor = pf_sys

        else:
            # count number of components in failed state
            num_failed = jnp.sum(state == self.n_damage_states - 1)
            num_operational = self.n_components - num_failed

            # k-out-of-n:G system is functional if at least k out of n
            # components are operational
            is_failed = jnp.less(num_operational, self.k)

            factor = is_failed

        _penalty = self.failure_cost * factor

        return _penalty

    def calculate_reward(self, state, belief, action) -> float:
        action_rewards = self._compute_action_rewards(self.component_list, action).sum(
            axis=0
        )

        # System rewards
        state_rewards = self._compute_state_rewards(state.damage_state, belief)

        # Mobilisation reward
        mobilised = jnp.greater(jnp.sum(action), 0)
        mobilisation_reward = mobilised * self.mobilisation_reward

        reward = action_rewards + state_rewards + mobilisation_reward
        return reward

    def calculate_returns(self, state, reward):
        return state.episode_return + reward * self.discount_factor**state.timestep

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

        # split keys into keys for damage transitions and observations
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
        reward = self.calculate_reward(state, belief, action)

        # returns
        returns = self.calculate_returns(state, reward)

        timestep = state.timestep + 1

        terminated = False
        truncated = self.is_truncated(timestep)
        done = jnp.logical_or(terminated, truncated)

        # info
        info = {"returns": returns}

        next_state = EnvState(
            damage_state=next_damage_state,
            observation=observation,
            belief=belief,
            timestep=timestep,
            episode_return=returns * jnp.logical_not(done),
        )

        return self.get_obs(next_state), next_state, reward, terminated, truncated, info

    def step(
        self, key, state, action
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:

        key, subkeys = self.split_key(key)

        # environment step
        obs_st, state_st, reward, terminated, truncated, info = self.step_env(
            subkeys, state, action
        )

        # reset env
        key, sub_key = jax.random.split(key)
        obs_re, state_re = self.reset(sub_key)

        # Auto-reset environment based on termination/truncation
        state = jax.tree.map(
            lambda x, y: jax.lax.select(truncated, x, y), state_re, state_st
        )

        obs = jax.tree.map(lambda x, y: jax.lax.select(truncated, x, y), obs_re, obs_st)

        return obs, state, reward, terminated, truncated, info

    @partial(vmap, in_axes=(None, 0, None))
    def sample(self, key: chex.PRNGKey, dist: chex.Array) -> chex.Array:
        return jax.random.choice(key, self.n_damage_states, p=dist)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:

        # duplicate the initial belief for each component
        belief = np.tile(self.initial_belief, (self.n_components, 1))
        belief = jnp.array(belief)

        # sample damage state
        subkeys = jax.random.split(key, 1 + self.n_components)
        damage_state = self.sample(subkeys[1:], self.initial_belief)

        # sample observation
        subkeys = jax.random.split(subkeys[0], 1 + self.n_components)
        observation = self.sample(subkeys[1:], self.initial_belief)

        env_state = EnvState(
            damage_state=damage_state,
            observation=observation,
            belief=belief,
        )
        return self.get_obs(env_state), env_state

    def get_obs(self, state: EnvState) -> chex.Array:

        if self.wrapper == "OneHot":
            local_obs = jax.nn.one_hot(state.damage_state, self.n_damage_states)

        elif self.wrapper == "Obs":
            local_obs = state.observation

        elif self.wrapper == "Filter":
            local_obs = state.belief

        return local_obs

    def is_truncated(self, timestep: float) -> bool:
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
