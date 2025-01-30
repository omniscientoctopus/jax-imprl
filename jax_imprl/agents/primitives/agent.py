from functools import partial

import chex
import jax
import optax

import jax.numpy as jnp
import flashbax as fbx

from jax_imprl.agents.modules.schedulers import LinearScheduler


@chex.dataclass(frozen=True)
class TransitionTuple:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array


class Agent:

    def __init__(
        self,
        env,
        config,
        experiment_config,
        eval_env=None,
        checkpoint_path=None,
    ):
        self.env = env
        if eval_env is None:
            self.eval_env = env
        else:
            self.eval_env = eval_env
        self.config = config
        self.experiment_config = experiment_config

        self.to_num_timesteps = lambda episodes: episodes * self.env.env.time_horizon

        # logging and evaluation
        self.num_episodes = experiment_config["NUM_EPISODES"]
        self.logging_freq = experiment_config["LOGGING_FREQ"]
        self.eval_freq = experiment_config["EVAL_FREQ"]
        self.eval_episodes = experiment_config["EVAL_EPISODES"]
        self.train_timesteps = self.to_num_timesteps(self.num_episodes)
        self.eval_timesteps = self.eval_episodes * self.eval_env.env.time_horizon

        # checkpointing
        self.checkpoint_path = checkpoint_path

        self.discount_factor = config["DISCOUNT_FACTOR"]

        # initialize exploration scheduler
        self.exploration_scheduler = LinearScheduler(
            config["EXPLORATION_STRATEGY"]["final_value"],
            steps=self.to_num_timesteps(config["EXPLORATION_STRATEGY"]["total_iters"]),
            initial=config["EXPLORATION_STRATEGY"]["initial_value"],
        )

        # initialize replay buffer
        self.replay_buffer = fbx.make_flat_buffer(
            max_length=config["MAX_MEMORY_SIZE"],
            min_length=config["BATCH_SIZE"],
            sample_batch_size=config["BATCH_SIZE"],
            add_sequences=True,
            add_batch_size=None,
        )

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0))
    def train(self, key):

        # Initialize the runner
        runner = self.init_runner(key)

        # Primary training loop
        runner, metrics = jax.lax.scan(
            self.update_runner, runner, length=self.train_timesteps
        )

        # Summarize metrics over episodes
        metrics = self.summarize_metrics(metrics)

        return runner, metrics

    def compute_loss(self, *args):
        return jnp.mean(self.compute_per_sample_loss(*args), axis=0).squeeze()

    @partial(jax.jit, static_argnums=(0,))
    def evaluate_and_checkpoint(self, main_runner, metrics):

        # Initialize the runner
        eval_runner = self.init_eval_runner(main_runner)

        # Primary evaluation loop
        eval_runner, metrics = jax.lax.scan(
            self.update_eval_runner,
            eval_runner,
            length=self.eval_timesteps,
        )

        main_runner = main_runner.replace(key=eval_runner.key)

        # compute aggregate metrics (avoiding indexing)
        evals = metrics["returns"] * metrics["dones"]
        evals = evals.reshape(self.eval_episodes, -1).sum(axis=1)
        eval_mean = -jnp.mean(evals)

        # checkpoint model
        jax.debug.callback(self.save_checkpoint, main_runner)

        metrics = {"eval_mean": eval_mean}

        return main_runner, metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def update_eval_runner(self, eval_runner, unused):

        # 1. Select action
        eval_runner, action = self.select_action(eval_runner)

        # 2. Environment step
        key, step_key = jax.random.split(eval_runner.key)
        next_obs, env_state, reward, terminated, truncated, info = self.eval_env.step(
            step_key, eval_runner.env_state, action
        )

        # 3. Update metrics
        metrics = {
            "returns": info["returns"],
            "dones": jnp.logical_or(terminated, truncated),
        }

        # 4. Update runner state
        eval_runner = eval_runner.replace(key=key, env_state=env_state, obs=next_obs)

        return eval_runner, metrics

    def summarize_metrics(self, metrics):
        # compute aggregate metrics over episodes
        dones = metrics["dones"]

        summary = {}
        for key, value in metrics.items():
            metric = value * dones
            summary[key] = metric.reshape(self.num_episodes, -1).sum(axis=1)

        return summary

    def is_time_to_evaluate(self, ep, done):

        cond_1 = lambda ep: (ep % self.eval_freq) == 0
        cond_2 = lambda ep: ep == (self.num_episodes - 1)

        c1 = jnp.logical_or(cond_1(ep), cond_2(ep))

        return jax.lax.select(c1, done, False)

    def init_lr_scheduler(self, config):

        if config["lr_scheduler"] == "linear":
            transition_steps = self.to_num_timesteps(config["lr_total_iters"])

            assert (
                transition_steps <= self.train_timesteps
            ), "Transition steps must be less than total timesteps"
            lr_scheduler = optax.schedules.linear_schedule(
                init_value=config["lr_initial"],
                end_value=config["lr_final"],
                transition_steps=transition_steps,
            )
        else:
            raise NotImplementedError

        return lr_scheduler

    @staticmethod
    def init_optimizer(config, lr_scheduler):

        if config["optimizer"] == "adam":
            optimizer = optax.adam(lr_scheduler)
        else:
            raise NotImplementedError

        return optimizer

    def init_environment(self, key):
        key, env_rng = jax.random.split(key, 2)
        init_obs, env_state = self.eval_env.reset(env_rng)
        return key, init_obs, env_state
