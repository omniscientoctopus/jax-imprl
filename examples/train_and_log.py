"""
A simple example to log the training process using wandb.

Typically, wandb is used to log metrics during training, but doing this 
in JAX requires returning control to python (via jax callbacks) 
which is extremely slow. To avoid this, we log metrics at the end of
training. The downside is that we can't see the training process in
real-time, but this is the best workaround for now.

Remarks: 
- Currently checkpointing is supported only for num_runs = 1.
"""

import os
import math
import yaml
import string, random
from datetime import datetime

import jax
import wandb
import numpy as np
import matplotlib.pyplot as plt

import jax_imprl.envs
from jax_imprl.agents.DDQN import DDQN

os.environ["WANDB__SERVICE_WAIT"] = "300"

print(f"Devices: {jax.devices()}")


def get_experiment_config():
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, "configs", "experiment_config.yaml")
    with open(config_path, "r") as f:
        experiment_config = yaml.safe_load(f)

    assert (
        experiment_config["EVAL_FREQ"] % experiment_config["LOGGING_FREQ"] == 0
    ), "EVAL_FREQ should be a multiple of LOGGING_FREQ, else eval episodes will be missed."

    return experiment_config


def create_experiment(checkpoint=True):
    cwd = os.getcwd()
    random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_name = f"{timestamp}-{random_string}"
    if checkpoint:
        checkpoint_path = os.path.join(cwd, "model_checkpoints", experiment_name)
        return experiment_name, checkpoint_path
    return experiment_name, None


def get_agent_configs():
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, "configs", "DDQN.yaml")
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def get_envs(env_name, env_setting, env_kwargs):
    env = jax_imprl.envs.make(
        env_name, env_setting, single_agent=True, **env_kwargs
    )
    eval_env = jax_imprl.envs.make(
        env_name, env_setting, single_agent=True, eval_env=True
    )

    return env, eval_env


def plot_results(experiment_config, logger):
    env_setting = experiment_config["ENV_SETTING"]
    experiment_name = experiment_config["EXPERIMENT_NAME"]
    num_episodes = experiment_config["NUM_EPISODES"]
    eval_freq = experiment_config["EVAL_FREQ"]
    eval_episodes = list(range(0, num_episodes, eval_freq)) + [num_episodes - 1]
    x = np.array(eval_episodes)
    y = logger["eval_mean"][:, x].T

    print(f"Best: {y.min(axis=0)}")

    plt.plot(x, y, "o-")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Cost")
    plt.title(f"{env_setting}")
    plt.grid()
    plt.savefig(f"{experiment_name}-{env_setting}.png")
    return num_episodes


def log_to_wandb(experiment_config, agent_config, logger):
    wandb_config = {
        "experiment_config": experiment_config,
        "agent_config": agent_config,
    }

    num_episodes = experiment_config["NUM_EPISODES"]
    num_runs = experiment_config["NUM_RUNS"]
    logging_freq = experiment_config["LOGGING_FREQ"]
    eval_freq = experiment_config["EVAL_FREQ"]
    eval_episodes = list(range(0, num_episodes, eval_freq)) + [num_episodes - 1]
    logged_episodes = list(range(0, num_episodes, logging_freq)) + [num_episodes - 1]

    # Loop over runs
    for run in range(num_runs):
        wandb_run = wandb.init(
            project=experiment_config["WANDB_PROJECT"],
            entity=experiment_config["WANDB_ENTITY"],
            config=wandb_config,
        )

        best_cost, best_checkpt = math.inf, 0
        # Loop over all the episodes at logging frequency
        for ep in logged_episodes:
            # Add all the metrics
            _log = {key: value[run, ep] for key, value in logger.items()}

            # Add eval episodes
            if ep in eval_episodes:
                eval_mean = _log["eval_mean"]
                if eval_mean < best_cost:
                    best_cost, best_checkpt = eval_mean, ep
                _log.update({"eval_ep": ep})

            wandb.log(_log, step=ep)

        wandb.run.summary["best_cost"] = best_cost
        wandb.run.summary["best_checkpt"] = best_checkpt

        wandb_run.finish()


if __name__ == "__main__":
    # Experiment
    experiment_config = get_experiment_config()

    # Environment
    env_name = experiment_config["ENV_NAME"]
    env_setting = experiment_config["ENV_SETTING"]
    env_kwargs = experiment_config["ENV_KWARGS"]
    env, eval_env = get_envs(env_name, env_setting, env_kwargs)

    # Agent
    experiment_name, checkpoint_path = create_experiment(checkpoint=False)
    experiment_config["EXPERIMENT_NAME"] = experiment_name
    agent_config = get_agent_configs()[env_name]
    agent = DDQN(
        env,
        agent_config,
        experiment_config,
        eval_env=eval_env,
        checkpoint_path=checkpoint_path,
    )

    # Seeds
    seed = experiment_config["SEED"]
    num_runs = experiment_config["NUM_RUNS"]
    key = jax.random.PRNGKey(seed)
    subkeys = jax.random.split(key, num_runs)

    # Training
    runner, metrics = jax.block_until_ready(agent.train(subkeys))

    # Logging
    plot_results(experiment_config, metrics)

    if experiment_config["WANDB"]:
        log_to_wandb(experiment_config, agent_config, metrics)
