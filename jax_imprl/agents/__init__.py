import jax_imprl.agents

from jax_imprl.agents.DDQN import DDQN

def get_agent_class(algorithm):
    try:
        return getattr(jax_imprl.agents, algorithm)
    except AttributeError:
        raise NotImplementedError(f"The algorithm '{algorithm}' is not implemented.")