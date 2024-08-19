# Module to make structural environments

import os
import yaml

from jax_imprl.structural_envs.jax_k_out_of_n import JaxKOutOfN


def make(setting=None, **kwargs):
    """
    Make a structural environment.
    """
    # get the environment module
    env_class = JaxKOutOfN

    # get the environment config
    pwd = os.path.dirname(__file__)
    rel_path = f"env_configs/{setting}.yaml"
    abs_file_path = os.path.join(pwd, rel_path)

    with open(abs_file_path) as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)

    # get baselines
    rel_path = "baselines.yaml"
    abs_file_path = os.path.join(pwd, rel_path)

    with open(abs_file_path) as file:
        all_baselines = yaml.load(file, Loader=yaml.FullLoader)

    baselines = all_baselines["k_out_of_n"][setting]

    # create the environment
    env = env_class(env_config, baselines=baselines, **kwargs)

    return env
