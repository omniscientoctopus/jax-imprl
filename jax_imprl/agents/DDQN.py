from functools import partial
from typing import Any

import chex
import flax
import jax
import jax.numpy as jnp
import orbax
import optax
import flashbax as fbx
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from jax_imprl import MLP, Agent, EvalRunnerState, TransitionTuple

# jax.config.update("jax_disable_jit", True)


@chex.dataclass(frozen=True)
class TransitionTuple:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    next_obs: chex.Array


@flax.struct.dataclass
class Runner:
    # Q-learning
    QState: TrainState
    target_network_params: flax.core.FrozenDict

    # defaults
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array
    buffer_state: Any
    ep: int = 0
    total_timesteps: int = 0

    def get_eps(self, exploration_scheduler):
        return exploration_scheduler.get(self.total_timesteps)


@flax.struct.dataclass
class EvalRunner:
    # Q-learning
    QState: TrainState

    # defaults
    key: chex.PRNGKey
    env_state: Any
    obs: chex.Array

    def get_eps(self, exploration_scheduler):
        return 0.0  # Always greedy


class DDQN:
    name = "DDQN"
    full_name = "Double Deep Q-Network"

    def __init__(
        self, env, config, experiment_config, eval_env=None, checkpoint_path=None
    ):

        ############ common setup ############

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

        ############ agent-specific setup ############

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
            add_sequences=True,  # TODO: setting to False throws an error?!
            add_batch_size=None,
        )

        # Initialize Q network
        _input = self.env.obs_dim
        _hidden = config["NETWORK_CONFIG"]["hidden_layers"]
        _output = self.env.action_space().n
        self.q_network = MLP([_input] + _hidden + [_output])

        # Initialize learning rate scheduler
        transition_steps = self.to_num_timesteps(
            config["NETWORK_CONFIG"]["lr_total_iters"]
        )
        self.lr_scheduler = optax.schedules.linear_schedule(
            init_value=config["NETWORK_CONFIG"]["lr_initial"],
            end_value=config["NETWORK_CONFIG"]["lr_final"],
            transition_steps=transition_steps,
        )

        # Initialize optimizer
        self.optimizer = optax.adam(self.lr_scheduler)

    @partial(jax.jit, static_argnums=(0,))
    def init_runner(self, key):

        # Initialize the Q-network parameters
        key, key_1, key_2 = jax.random.split(key, 3)

        obs, _ = self.env.reset(key_1)
        init_x = jnp.zeros_like(obs)
        q_network_params = self.q_network.init(key_2, init_x)

        q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=q_network_params,
            tx=self.optimizer,
        )

        # Initialize the target network parameters
        target_q_network_params = jax.tree.map(lambda x: jnp.copy(x), q_state.params)

        # Initialize environment
        key, reset_key = jax.random.split(key, 2)
        init_obs, env_state = self.env.reset(reset_key)

        # Initialize replay buffer
        key, subkey = jax.random.split(key, 2)
        dummy_action = self.env.action_space().sample(subkey)
        _experience = TransitionTuple(
            obs=jnp.reshape(init_obs, (1, -1)),
            action=dummy_action,
            reward=jnp.array([0.0], dtype=jnp.float32),
            terminated=jnp.reshape(False, (1, 1)),
            truncated=jnp.reshape(False, (1, 1)),
            next_obs=jnp.reshape(init_obs, (1, -1)),
        )
        buffer_state = self.replay_buffer.init(_experience)

        runner = Runner(
            QState=q_state,
            target_network_params=target_q_network_params,
            buffer_state=buffer_state,
            obs=init_obs,
            env_state=env_state,
            key=key,
        )

        return runner

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

    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, runner):

        ## if training: epsilon-greedy strategy
        #  else: greedy strategy
        eps = runner.get_eps(self.exploration_scheduler)

        # generate random number
        key, key_1, key_2 = jax.random.split(runner.key, 3)

        # Greedy action
        q_values = runner.QState.apply_fn(runner.QState.params, runner.obs)
        q_values = jax.lax.stop_gradient(q_values)
        greedy_action = jnp.argmax(q_values)

        # Random action
        random_action = jax.random.randint(
            key_1, (1,), 0, self.env.action_space().n, jnp.int32
        )

        # if random number < epsilon, select random action
        # else, select action from Q-network
        action = jnp.where(
            jax.random.uniform(key_2) < eps,
            random_action,
            greedy_action,
        )

        runner = runner.replace(key=key)

        return runner, action

    @partial(jax.jit, static_argnums=(0,))
    def update_replay_buffer(
        self, runner, obs, action, reward, terminated, truncated, next_obs
    ):
        experience = TransitionTuple(
            obs=jnp.reshape(obs, (1, -1)),
            action=action,
            reward=jnp.array([reward]),
            terminated=jnp.reshape(terminated, (1,)),
            truncated=jnp.reshape(truncated, (1,)),
            next_obs=jnp.reshape(next_obs, (1, -1)),
        )
        buffer_state = self.replay_buffer.add(runner.buffer_state, experience)

        return runner.replace(buffer_state=buffer_state)

    @partial(jax.jit, static_argnums=(0,))
    def learning_phase(self, runner, unused):

        # 1. Sample batch from replay buffer
        key, subkey = jax.random.split(runner.key)
        batch = self.replay_buffer.sample(runner.buffer_state, subkey).experience
        args = (
            batch.first.obs.squeeze(),
            batch.first.action,
            batch.first.reward,
            batch.first.terminated.reshape(-1, 1),
            batch.first.truncated.reshape(-1, 1),
            batch.first.next_obs.squeeze(),
        )

        # 2. Compute loss (per sample)
        @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
        def compute_per_sample_loss(
            q_network_params,
            target_network_params,
            obs,
            action,
            reward,
            terminated,
            truncated,
            next_obs,
        ):
            # compute Q-values
            all_q_values = self.q_network.apply(q_network_params, obs)
            q_value = all_q_values[action]

            # compute future value (using target network)
            all_q_next_values = self.q_network.apply(q_network_params, next_obs)
            best_action = jnp.argmax(all_q_next_values)
            all_q_target_next_values = self.q_network.apply(
                target_network_params, next_obs
            )
            future_value = all_q_target_next_values[best_action]

            # compute target
            _mask = jnp.where(terminated, 1, 0)
            target = reward + (1 - _mask) * self.discount_factor * future_value
            target = jax.lax.stop_gradient(target)

            return (q_value - target) ** 2

        loss_fn = lambda *args: jnp.mean(
            compute_per_sample_loss(*args), axis=0
        ).squeeze()

        # 3. Compute loss (batch)
        td_loss, grads = jax.value_and_grad(loss_fn)(
            runner.QState.params, runner.target_network_params, *args
        )

        # 4. Update Q-network
        q_state = runner.QState.apply_gradients(grads=grads)
        runner = runner.replace(key=key, QState=q_state)

        metrics = {"td_loss": td_loss}

        return runner, metrics

    @partial(jax.jit, static_argnums=(0, 2))
    def update_runner(self, runner, unused):

        # 1. select action
        runner, action = self.select_action(runner)

        # 2. environment
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, terminated, truncated, info = self.env.step(
            step_key, runner.env_state, action
        )

        # see: https://instadeepai.github.io/flashbax/#sequential-data-addition
        # auto-reset over-writes the next_obs and env_state, this
        # problematic when going from episode n to episode n+1. Fix by
        # extracting truncated/terminated obs from info.
        episode_return = info["returns"]
        _next_obs = info["next_obs"]
        ep_done = jnp.logical_or(terminated, truncated)

        # 3. update replay buffer
        runner = self.update_replay_buffer(
            runner, runner.obs, action, reward, terminated, truncated, _next_obs
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
            {"td_loss": 0.0},
        )

        # 5. update target network
        def update_target_network(runner):
            # hard update
            params = jax.tree.map(lambda x: jnp.copy(x), runner.QState.params)
            return runner.replace(target_network_params=params)

        runner = jax.lax.cond(
            runner.ep % self.config["TARGET_UPDATE_FREQ"] == 0,
            update_target_network,
            lambda x: x,
            runner,
        )

        # 6. evaluate
        # episode is done *and* either time to evaluate *or* last episode
        runner, eval_metrics = jax.lax.cond(
            jnp.logical_and(
                ep_done,
                jnp.logical_or(
                    (runner.ep % self.eval_freq) == 0,
                    runner.ep == (self.num_episodes - 1),
                ),
            ),
            self.evaluate_and_checkpoint,
            lambda *x: x,
            runner,
            {"eval_mean": 0.0},
        )

        # 7. compute metrics
        metrics = {
            "episode": runner.ep,
            "total_timesteps": runner.total_timesteps,
            "returns": -episode_return,
            "lr": self.lr_scheduler(runner.total_timesteps),
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
    def evaluate_and_checkpoint(self, main_runner, metrics):

        # 1. Initialize the runner
        key, reset_key = jax.random.split(main_runner.key, 2)
        init_obs, env_state = self.eval_env.reset(reset_key)
        eval_runner = EvalRunner(
            QState=main_runner.QState,
            env_state=env_state,
            obs=init_obs,
            key=key,
        )

        # 2. Define update function
        # @partial(jax.jit, static_argnums=(0, 2))
        def update_eval_runner(runner, unused=None):

            # 1. Select action
            runner, action = self.select_action(runner)

            # 2. Environment step
            key, step_key = jax.random.split(runner.key)
            next_obs, env_state, reward, terminated, truncated, info = (
                self.eval_env.step(step_key, runner.env_state, action)
            )

            # 3. Update metrics
            metrics = {
                "returns": info["returns"],
                "dones": jnp.logical_or(terminated, truncated),
            }

            # 4. Update runner state
            runner = runner.replace(key=key, env_state=env_state, obs=next_obs)

            return runner, metrics

        # 3. Primary evaluation loop
        eval_runner, metrics = jax.lax.scan(
            update_eval_runner,
            eval_runner,
            length=self.eval_timesteps,
        )

        main_runner = main_runner.replace(key=eval_runner.key)

        # 4. compute aggregate metrics (avoiding indexing)
        evals = metrics["returns"] * metrics["dones"]
        evals = evals.reshape(self.eval_episodes, -1).sum(axis=1)
        eval_mean = -jnp.mean(evals)

        # 5. checkpoint model
        jax.debug.callback(self.save_checkpoint, main_runner)

        metrics = {"eval_mean": eval_mean}

        return main_runner, metrics

    def summarize_metrics(self, metrics):
        # compute aggregate metrics over episodes
        dones = metrics["dones"]

        summary = {}
        for key, value in metrics.items():
            metric = value * dones
            summary[key] = metric.reshape(self.num_episodes, -1).sum(axis=1)

        return summary

    def save_checkpoint(self, runner):

        if self.checkpoint_path is not None:

            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

            # create checkpoint
            ckpt = runner.QState

            path = f"{self.checkpoint_path}/chkpt_{runner.ep}"

            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(path, ckpt, save_args=save_args)

    def load_checkpoint(self, runner, path, ep):

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        path = f"{path}/chkpt_{ep}"

        restored_state = orbax_checkpointer.restore(path)

        runner = runner.replace(
            params=restored_state["params"], tx=restored_state["opt_state"]
        )

        return runner
