# jax-imprl
A JAX accelerated version of IMPRL (Inspection and Maintenance Planning with Reinforcement Learning), a library for applying reinforcement learning to inspection and maintenance planning of deteriorating engineering systems.

## Installation

### 1. Create a virtual environment

```bash
conda create --name jax_imprl_env -y python==3.9
conda activate jax_imprl_env
```

### 2. Install the dependencies

```bash
pip install poetry==1.8 # or conda install -c conda-forge poetry==1.8
poetry install
```


<details>
<summary>Installing additional packages</summary>

You can them add via `poetry add` ([official docs](https://python-poetry.org/docs/cli/#add)) in the command line. 

For example, to install [Jupyter notebook](https://pypi.org/project/notebook/),

```bash 
# Allow >=7.1.2, <8.0.0 versions
poetry add notebook@^7.1.2
```
This will resolve the package dependencies (and adjust versions of transitive dependencies if necessary) and install the package. If the package dependency cannot be resolved, try to relax the package version and try again.
</details>

### 3. Setup wandb

For logging, the library relies on [wandb](https://wandb.ai). You can log into wandb using your private API key, 

```bash
wandb login
# <enter wandb API key>
```

## Related Work

- [IMPRL]()

- [IMP-MARL](https://github.com/moratodpg/imp_marl): a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.

    - Environments: (Correlated and uncorrelated) k-out-of-n systems and offshore wind structural systems.
    - RL solvers: Provides wrappers for interfacing with several (MA)RL libraries such as [EPyMARL](https://github.com/uoe-agents/epymarl), [Rllib](imp_marl/imp_wrappers/examples/rllib/rllib_example.py), [MARLlib](imp_marl/imp_wrappers/marllib/marllib_wrap_ma_struct.py) etc.