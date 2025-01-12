import jax
import jax.numpy as jnp
from functools import partial

import flax
import orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from jax_imprl import MLP, Agent, TransitionTuple, RunnerState, EvalRunnerState


class RunnerState(RunnerState):
    QState: TrainState
    target_network_params: flax.core.FrozenDict


class EvalRunnerState(EvalRunnerState):
    QState: TrainState


class DDQN(Agent):
    name = "DDQN"
    full_name = "Double Deep Q-Network"

    def __init__(
        self, env, config, experiment_config, eval_env=None, checkpoint_path=None
    ):

        super().__init__(env, config, experiment_config, eval_env, checkpoint_path)

        # Initialize Q network
        _input = self.env.obs_dim
        _hidden = config["NETWORK_CONFIG"]["hidden_layers"]
        _output = self.env.action_space().n
        self.q_network = MLP([_input] + _hidden + [_output])

        # Initialize learning rate scheduler
        self.lr_scheduler = self.init_lr_scheduler(
            config["NETWORK_CONFIG"]
        )

        # Initialize optimizer
        self.optimizer = self.init_optimizer(
            config["NETWORK_CONFIG"], self.lr_scheduler
        )

    def init_q_state(self, key):

        key, key_1, key_2 = jax.random.split(key, 3)

        obs, _ = self.env.reset(key_1)
        init_x = jnp.zeros_like(obs)
        q_network_params = self.q_network.init(key_2, init_x)

        q_state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=q_network_params,
            tx=self.optimizer,
        )

        return key, q_state

    def init_replay_buffer(self, key, init_obs):
        key, subkey = jax.random.split(key, 2)
        dummy_action = self.env.action_space().sample(subkey)

        _experience = TransitionTuple(
            obs=jnp.reshape(init_obs, (1, -1)),
            action=dummy_action,
            reward=jnp.array([0.0], dtype=jnp.float32),
            terminated=jnp.reshape(False, (1, 1)),
            truncated=jnp.reshape(False, (1, 1)),
        )
        buffer_state = self.replay_buffer.init(_experience)
        return key, buffer_state

    @partial(jax.jit, static_argnums=(0,))
    def get_random_action(self, key):
        return jax.random.randint(key, (1,), 0, self.env.action_space().n, jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def get_greedy_action(self, runner):
        q_values = runner.QState.apply_fn(runner.QState.params, runner.obs)
        q_values = jax.lax.stop_gradient(q_values)
        greedy_action = jnp.argmax(q_values)
        return greedy_action

    @partial(jax.jit, static_argnums=(0,))
    def select_action(self, runner):

        ## if training: epsilon-greedy strategy
        #  else: greedy strategy
        eps = runner.get_eps(self.exploration_scheduler)

        # generate random number
        key, key_1, key_2 = jax.random.split(runner.key, 3)

        # if random number < epsilon, select random action
        # else, select action from Q-network
        action = jnp.where(
            jax.random.uniform(key_2) < eps,
            self.get_random_action(key_1),
            self.get_greedy_action(runner),
        )

        runner = runner.replace(key=key)

        return runner, action

    @partial(jax.jit, static_argnums=(0,))
    def update_replay_buffer(self, runner, obs, action, reward, terminated, truncated):

        experience = TransitionTuple(
            obs=jnp.reshape(obs, (1, -1)),
            action=action,
            reward=jnp.array([reward]),
            terminated=jnp.reshape(terminated, (1, 1)),
            truncated=jnp.reshape(truncated, (1, 1)),
        )
        buffer_state = self.replay_buffer.add(runner.buffer_state, experience)

        return runner.replace(buffer_state=buffer_state)

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0))
    def compute_per_sample_loss(
        self,
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

        # compute next Q-values
        all_q_next_values = self.q_network.apply(
            jax.lax.stop_gradient(q_network_params), next_obs
        )
        best_action = jnp.argmax(all_q_next_values)
        all_q_target_next_values = self.q_network.apply(
            jax.lax.stop_gradient(target_network_params), next_obs
        )
        q_next_value = all_q_target_next_values[best_action]

        # compute target
        _mask = jnp.where(terminated, 1, 0)
        target = reward + (1 - _mask) * self.discount_factor * q_next_value

        return (q_value - target) ** 2

    @partial(jax.jit, static_argnums=(0,))
    def learning_phase(self, runner, unused):

        # sample batch from replay buffer
        key, subkey = jax.random.split(runner.key)
        batch = self.replay_buffer.sample(runner.buffer_state, subkey).experience
        _obs, _action, _reward, _terminated, _truncated, _next_obs = (
            batch.first.obs.squeeze(),
            batch.first.action,
            batch.first.reward,
            batch.first.terminated.reshape(-1, 1),
            batch.first.truncated.reshape(-1, 1),
            batch.second.obs.squeeze(),
        )

        # compute loss
        td_loss, grads = jax.value_and_grad(self.compute_loss)(
            runner.QState.params,
            runner.target_network_params,
            _obs,
            _action,
            _reward,
            _terminated,
            _truncated,
            _next_obs,
        )

        # update
        q_state = runner.QState.apply_gradients(grads=grads)
        runner = runner.replace(key=key, QState=q_state)

        metrics = {"td_loss": td_loss}

        return runner, metrics

    @partial(jax.jit, static_argnums=(0,))
    def update_target_network(self, runner):

        # Hard update
        target_network_params = jax.tree_map(
            lambda x: jnp.copy(x), runner.QState.params
        )

        return runner.replace(target_network_params=target_network_params)

    @partial(jax.jit, static_argnums=(0, 2))
    def update_runner(self, runner, unused):

        # 1. select action
        runner, action = self.select_action(runner)

        # 2. environment
        key, step_key = jax.random.split(runner.key)
        next_obs, env_state, reward, terminated, truncated, info = self.env.step(
            step_key, runner.env_state, action
        )

        episode_return = info["returns"]
        ep_done = jnp.logical_or(terminated, truncated)

        # 3. update replay buffer
        runner = self.update_replay_buffer(
            runner, runner.obs, action, reward, terminated, truncated
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
        runner = jax.lax.cond(
            runner.ep % self.config["TARGET_UPDATE_FREQ"] == 0,
            self.update_target_network,
            lambda x: x,
            runner,
        )

        # 7. evaluate
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
    def init_runner(self, key):

        # Initialize the Q-network parameters
        key, q_state = self.init_q_state(key)

        # Initialize the target network parameters
        target_q_network_params = jax.tree.map(lambda x: jnp.copy(x), q_state.params)

        # Initialize environment
        key, init_obs, env_state = self.init_environment(key)

        # Initialize replay buffer
        key, buffer_state = self.init_replay_buffer(key, init_obs)

        runner = RunnerState(
            QState=q_state,
            target_network_params=target_q_network_params,
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
        eval_runner = EvalRunnerState(
            QState=main_runner.QState,
            env_state=env_state,
            obs=init_obs,
            key=key,
        )

        return eval_runner

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
