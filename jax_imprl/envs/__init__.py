# Module to make structural environments

import os
import yaml


def make(name, setting=None, single_agent=True, **kwargs):
    """
    Make a structural environment.
    """

    if name in ["k_out_of_n", "k_out_of_n_infinite"]:

        import jax_imprl.envs.structural_envs.jax_k_out_of_n
        import jax_imprl.envs.structural_envs.jax_k_out_of_n_infinite
        from jax_imprl.envs.structural_envs.single_agent_wrapper import (
            SingleAgentWrapper,
        )

        # get the environment module
        module = getattr(jax_imprl.envs.structural_envs, f"jax_{name}")
        env_class = getattr(module, "JaxKOutOfN")

        rel_path_config = f"structural_envs/env_configs/{setting}.yaml"
        rel_path_baselines = f"structural_envs/baselines.yaml"

    elif name == "matrix_game":

        import jax_imprl.envs.game_envs.jax_matrix_game

        # get class MatrixGame
        module = getattr(jax_imprl.envs.game_envs, name)
        env_class = getattr(module, "MatrixGame")

        rel_path_config = f"game_envs/env_configs/{setting}.yaml"
        rel_path_baselines = f"game_envs/baselines.yaml"

    pwd = os.path.dirname(__file__)

    # get the environment config
    env_config_path = os.path.join(pwd, rel_path_config)
    with open(env_config_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    # get the baselines for the environment
    baselines_path = os.path.join(pwd, rel_path_baselines)
    with open(baselines_path) as file:
        all_baselines = yaml.load(file, Loader=yaml.FullLoader)
    baselines = all_baselines[name][setting]

    # create the environment
    env = env_class(env_config, baselines=baselines, **kwargs)

    # wrap the environment
    if single_agent:
        env = SingleAgentWrapper(env)

    return env
