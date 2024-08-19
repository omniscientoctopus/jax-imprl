import jax.numpy as jnp
import flashbax as fbx
import optax

import wandb

from jax_imprl.agents.modules.schedulers import LinearScheduler


class Agent:

    def __init__(self, env, config, project_name=None, entity=None):
        self.env = env
        self.config = config

        self.config = config
        self.discount_factor = config["DISCOUNT_FACTOR"]

        # initialize exploration scheduler
        self.exploration_scheduler = LinearScheduler(
            config["EXPLORATION_STRATEGY"]["final_value"],
            num_episodes=config["EXPLORATION_STRATEGY"]["num_steps"],
            initial=config["EXPLORATION_STRATEGY"]["initial_value"],
        )

        # initialize learning rate scheduler
        self.learning_rate_scheduler = LinearScheduler(
            config["NETWORK_CONFIG"]["lr_final"],
            num_episodes=config["NETWORK_CONFIG"]["lr_total_iters"],
            initial=config["NETWORK_CONFIG"]["lr_initial"],
        )

        # initialize replay buffer
        self.replay_buffer = fbx.make_flat_buffer(
            max_length=config["MAX_MEMORY_SIZE"],
            min_length=config["BATCH_SIZE"],
            sample_batch_size=config["BATCH_SIZE"],
            add_sequences=True,
            add_batch_size=None,
        )

        # initialize optimizer
        self.optimizer = optax.adam(learning_rate=self.learning_rate_scheduler.get)

        # initialize WandB
        self.wandb = False
        if project_name is not None:
            wandb.init(project=project_name, entity=entity, config=config)

            self.wandb = True

    def compute_loss(self, *args):
        return jnp.mean(self.compute_per_sample_loss(*args), axis=0).squeeze()
