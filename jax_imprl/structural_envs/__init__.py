# Module to make structural environments

import os
import yaml

from jax_imprl.structural_envs.single_agent_wrapper import SingleAgentWrapper


def make(name, setting=None, single_agent=True, **kwargs):
    """
    Make a structural environment.
    """

    if name in [
        "k_out_of_n",
        "k_out_of_n_v3",
        "k_out_of_n_infinite",
        "k_out_of_n_shaping",
    ]:

        import jax_imprl.structural_envs.jax_k_out_of_n
        import jax_imprl.structural_envs.jax_k_out_of_n_infinite

        # get the environment module
        module = getattr(jax_imprl.structural_envs, f"jax_{name}")
        env_class = getattr(module, "JaxKOutOfN")

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

        baselines = all_baselines[name][setting]

        # create the environment
        env = env_class(env_config, baselines=baselines, **kwargs)

        # wrap the environment
        if single_agent:
            env = SingleAgentWrapper(env)

        return env

    else:
        raise ValueError(f"Unknown environment: {name}")
