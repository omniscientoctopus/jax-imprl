from typing import Any

import chex
import jax
import jax.numpy as jnp
from functools import partial

import flax
import orbax
import distrax
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from jax_imprl import MLP, Agent

# jax.config.update("jax_disable_jit", True)


@chex.dataclass(frozen=True)
class TransitionTuple:
    obs: chex.Array
    action: chex.Array
    action_log_prob: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array


@flax.struct.dataclass
class Runner:
    # Actor and Critic states
    ActorState: TrainState
    CriticState: TrainState

    # defaults
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array
    buffer_state: Any
    ep: int = 0
    total_timesteps: int = 0

    def get_eps(self, exploration_scheduler):
        return exploration_scheduler.get(self.total_timesteps)

    def get_lr(self, scheduler):
        return scheduler.get(self.total_timesteps)


@flax.struct.dataclass
class EvalRunner:
    # Actor State
    ActorState: TrainState

    # defaults
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array

    def get_eps(self, exploration_scheduler):
        return 0.0  # Always greedy


class JointActorCritic(Agent):
    name = "JAC"
    full_name = "Joint Actor-Critic"

    def __init__(
        self, env, config, experiment_config, eval_env=None, checkpoint_path=None
    ):

        super().__init__(env, config, experiment_config, eval_env, checkpoint_path)

        ############ Actor ############
        # Initialize actor network
        hidden_layers = self.config["ACTOR_CONFIG"]["hidden_layers"]
        input_dim = self.env.obs_dim
        output_dim = self.env.action_space().n
        self.actor = MLP([input_dim] + hidden_layers + [output_dim])

        # Initialize learning rate scheduler
        self.actor_lr_scheduler = self.init_lr_scheduler(self.config["ACTOR_CONFIG"])

        # Initialize optimizer
        self.actor_optimizer = self.init_optimizer(
            config["ACTOR_CONFIG"], self.actor_lr_scheduler
        )

        ############ Critic ############
        # Initialize critic network
        output_dim = 1
        hidden_layers = self.config["CRITIC_CONFIG"]["hidden_layers"]
        self.critic = MLP([input_dim] + hidden_layers + [output_dim])

        # Initialize learning rate scheduler
        self.critic_lr_scheduler = self.init_lr_scheduler(self.config["CRITIC_CONFIG"])

        # Initialize optimizer
        self.critic_optimizer = self.init_optimizer(
            config["CRITIC_CONFIG"], self.critic_lr_scheduler
        )

    def init_actor_and_critic_state(self, key):

        key, key_1, key_2, key_3 = jax.random.split(key, 4)
        obs, _ = self.env.reset(key_1)
        init_x = jnp.zeros_like(obs)

        actor_params = self.actor.init(key_2, init_x)
        critic_params = self.critic.init(key_3, init_x)

        actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=self.actor_optimizer
        )

        critic_state = TrainState.create(
            apply_fn=self.critic.apply, params=critic_params, tx=self.critic_optimizer
        )

        return key, actor_state, critic_state

    def init_replay_buffer(self, key, init_obs):
        key, subkey = jax.random.split(key, 2)
        dummy_action = self.env.action_space().sample(subkey)

        _experience = TransitionTuple(
            obs=jnp.reshape(init_obs, (1, -1)),
            action=dummy_action,
            action_log_prob=jnp.array([1.0], dtype=jnp.float32),
            reward=jnp.array([0.0], dtype=jnp.float32),
            terminated=jnp.reshape(False, (1, 1)),
            truncated=jnp.reshape(False, (1, 1)),
        )
        buffer_state = self.replay_buffer.init(_experience)
        return key, buffer_state

    @partial(jax.jit, static_argnums=(0,))
    def get_random_action(self, runner):
        num_actions = self.env.action_space().n
        logits = jnp.ones((num_actions,), dtype=jnp.float32) / 2
        return distrax.Categorical(logits=logits)

    @partial(jax.jit, static_argnums=(0,))
    def get_greedy_action(self, runner):
        logits = self.actor.apply(runner.ActorState.params, runner.obs)
        logits = jax.lax.stop_gradient(logits)
        return distrax.Categorical(logits=logits)

    @partial(jax.jit, static_argnums=(0))
    def select_action(self, runner):
        ## if training: epsilon-greedy strategy
        #  else: greedy strategy
        eps = runner.get_eps(self.exploration_scheduler)

        # generate random number
        key, key_1, key_2 = jax.random.split(runner.key, 3)

        # if random number < epsilon, select random action
        # else, select action from policy
        action_dist = jax.lax.cond(
            jax.random.uniform(key_1) < eps,
            self.get_random_action,
            self.get_greedy_action,
            runner,
        )

        action = action_dist.sample(seed=key_2, sample_shape=(1,))
        action_log_prob = action_dist.log_prob(action)

        runner = runner.replace(key=key)

        return runner, action, action_log_prob

    @partial(jax.jit, static_argnums=(0,))
    def update_replay_buffer(
        self, runner, obs, action, action_log_prob, reward, terminated, truncated
    ):
        experience = TransitionTuple(
            obs=jnp.reshape(obs, (1, -1)),
            action=action,
            action_log_prob=jnp.reshape(action_log_prob, (1, 1)),
            reward=jnp.array([reward]),
            terminated=jnp.reshape(terminated, (1, 1)),
            truncated=jnp.reshape(truncated, (1, 1)),
        )
        buffer_state = self.replay_buffer.add(runner.buffer_state, experience)

        return runner.replace(buffer_state=buffer_state)

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0))
    def compute_per_sample_loss(
        self,
        actor_params,
        critic_params,
        obs,
        action,
        action_log_prob,
        reward,
        terminated,
        truncated,
        next_obs,
    ):

        current_value = self.critic.apply(critic_params, obs)
        future_value = self.critic.apply(critic_params, next_obs)

        # compute target
        _mask = jnp.where(terminated, 1, 0)
        target = reward + (1 - _mask) * self.discount_factor * future_value

        # compute advantage
        advantage = target - current_value
        advantage = jax.lax.stop_gradient(advantage)

        # importance sampling weight
        logits = self.actor.apply(actor_params, obs)

        action_dist = distrax.Categorical(logits=logits)
        true_log_prob = action_dist.log_prob(action)
        weight = jnp.exp(true_log_prob - action_log_prob)
        # clip weights to reduce variance
        # weight = jnp.minimum(self.importance_sampling_weight_min, weight)
        weight = jnp.minimum(weight, 5.0)
        weight = jax.lax.stop_gradient(weight)

        # jax.debug.print("{x}, {y}", x=weight, y=true_log_prob)

        actor_loss = -true_log_prob * advantage * weight
        critic_loss = weight * (current_value - target) ** 2

        return actor_loss + critic_loss

    def compute_loss(self, *args):
        return jnp.mean(self.compute_per_sample_loss(*args), axis=0).squeeze()

    # Compute the L2 norm of gradients
    @staticmethod
    def l2_norm(tree):
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda acc, x: acc + jnp.sum(x**2), tree, initializer=0
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def learning_phase(self, runner, unused):

        # sample batch from replay buffer
        key, subkey = jax.random.split(runner.key)
        batch = self.replay_buffer.sample(runner.buffer_state, subkey).experience

        args = (
            batch.first.obs.squeeze(),
            batch.first.action,
            batch.first.action_log_prob,
            batch.first.reward,
            batch.first.terminated.reshape(-1, 1),
            batch.first.truncated.reshape(-1, 1),
            batch.second.obs.squeeze(),
        )

        loss, grads = jax.value_and_grad(
            self.compute_loss, argnums=(0, 1)
        )(runner.ActorState.params, runner.CriticState.params, *args)

        actor_grads, critic_grads = grads

        actor_state = runner.ActorState.apply_gradients(grads=actor_grads)
        critic_state = runner.CriticState.apply_gradients(grads=critic_grads)

        actor_grad_norm = self.l2_norm(actor_grads)
        critic_grad_norm = self.l2_norm(critic_grads)

        runner = runner.replace(
            key=key, ActorState=actor_state, CriticState=critic_state
        )

        metrics = {
            "actor_loss": loss,
            "critic_loss": loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }

        return runner, metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def update_runner(self, runner, unused):

        # 1. select action
        runner, action, action_log_prob = self.select_action(runner)

        # 2. environment
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, terminated, truncated, info = self.env.step(
            step_key, runner.env_state, action
        )

        episode_return = info["returns"]
        ep_done = jnp.logical_or(terminated, truncated)

        # 3. update replay buffer
        runner = self.update_replay_buffer(
            runner, runner.obs, action, action_log_prob, reward, terminated, truncated
        )

        # important: update obs after updating replay buffer
        runner = runner.replace(env_state=env_state, obs=next_obs, key=key)

        # 4. learning phase
        # check if buffer is ready for sampling
        runner, training_metrics = jax.lax.cond(
            self.replay_buffer.can_sample(runner.buffer_state),
            self.learning_phase,
            lambda *x: x,
            runner,
            {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
            },
        )

        # 5. evaluate
        runner, eval_metrics = jax.lax.cond(
            self.is_time_to_evaluate(runner.ep, ep_done),
            self.evaluate_and_checkpoint,
            lambda *x: x,
            runner,
            {"eval_mean": 0.0},
        )

        # 6. compute metrics
        metrics = {
            "episode": runner.ep,
            "total_timesteps": runner.total_timesteps,
            "returns": -episode_return,
            "lr_critic": self.critic_lr_scheduler(runner.total_timesteps),
            "lr_actor": self.actor_lr_scheduler(runner.total_timesteps),
            "eps": runner.get_eps(self.exploration_scheduler),
            "dones": ep_done,
            **training_metrics,
            **eval_metrics,
        }

        # if truncated/terminated: update episode count
        episode = jax.lax.cond(ep_done, lambda x: x + 1, lambda x: x, runner.ep)

        runner = runner.replace(total_timesteps=runner.total_timesteps + 1, ep=episode)

        return runner, metrics

    @partial(jax.jit, static_argnums=(0,))
    def init_runner(self, key):

        # Initialize actor and critic networks
        key, actor_state, critic_state = self.init_actor_and_critic_state(key)

        # Initialize environment
        key, init_obs, env_state = self.init_environment(key)

        # Initialize replay buffer
        key, buffer_state = self.init_replay_buffer(key, init_obs)

        runner = Runner(
            ActorState=actor_state,
            CriticState=critic_state,
            buffer_state=buffer_state,
            obs=init_obs,
            env_state=env_state,
            key=key,
        )

        return runner

    @partial(jax.jit, static_argnums=(0,))
    def init_eval_runner(self, main_runner):

        # Initialize the environment
        key, init_obs, env_state = self.init_environment(main_runner.key)

        # Initialize the runner
        eval_runner = EvalRunner(
            ActorState=main_runner.ActorState,
            env_state=env_state,
            obs=init_obs,
            key=key,
        )

        return eval_runner

    def save_checkpoint(self, runner):
        NotImplementedError

    def load_checkpoint(self, runner, path, ep):
        NotImplementedError
