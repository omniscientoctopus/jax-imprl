# jax-imprl 🚀
A JAX accelerated version of IMPRL (Inspection and Maintenance Planning with Reinforcement Learning), a library for applying reinforcement learning to inspection and maintenance planning of deteriorating engineering systems.

## Installation 📦

### 1. Install uv

<details>
<summary>Why install uv?</summary>
An extremely fast Python package and project manager, written in Rust. It is much faster than pip and pip-tools, and has a simple CLI for managing dependencies, virtual environments, and scripts. More info here: https://docs.astral.sh/uv/
</details>

You can install uv using the following methods ([see docs for OS-specific options](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
# macOS (Homebrew)
brew install uv

# Or via script (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a virtual environment

(Recommended) Create a uv-managed virtualenv:
```bash
uv venv --python 3.9 # create virtual environment
source .venv/bin/activate  # activate virtual environment
```

<details>
<summary>Alternative: conda</summary>

```bash
conda create --name jax_imprl_env -y python==3.9
conda activate jax_imprl_env
```

</details>

### 3. Install the dependencies (uv)

```bash
# Install base dependencies, creating uv.lock
uv sync

# Optional: include extras and/or dev tools
# Dev tools are in the "dev" dependency group
uv sync --group dev
```

<details>
<summary>Installing additional packages</summary>

Add packages with `uv add` and optionally assign them to an extra or group.

For example, to add [pandas](https://pypi.org/project/pandas/) allowing any 2.x release:

```bash
uv add "pandas>=2,<3"
```

To add a dev-only tool:

```bash
uv add --group dev ruff
```

If resolution fails, relax version ranges and retry.
</details>

### 4. (optional) Test the installation
You can run unit tests to verify that the installation was successful.

```bash
uv sync --group dev # ensure dev dependencies are installed
pytest -v tests
```

### 5. (optional) Setup wandb

For logging, the library relies on [wandb](https://wandb.ai). You can log into wandb using your private API key, 

```bash
wandb login
# <enter wandb API key>
```

## Docker 🐳

If you want to run the code in a containerized environment, you can use the following Docker image and the previous installation steps.

```bash
docker pull nvidia/cuda
```

In case you don't have accesss to NVIDIA GPUs, you can rent a cloud instance and load the above Docker image. For example, [vast.ai](https://vast.ai) at ~$0.30/hour ([pricing](https://vast.ai/#pricing))

```bash
https://cloud.vast.ai/?ref_id=113803&creator_id=113803&name=JAX%2BRL
```

## Related Work 🔗

- [IMPRL](https://github.com/omniscientoctopus/imprl): small-scale k-out-of-n environments with upto 5 components.

- [IMP-MARL](https://github.com/moratodpg/imp_marl): a platform for benchmarking the scalability of cooperative MARL methods in real-world engineering applications.

    - Environments: (Correlated and uncorrelated) k-out-of-n systems and offshore wind structural systems.
    - RL solvers: Provides wrappers for interfacing with several (MA)RL libraries such as [EPyMARL](https://github.com/uoe-agents/epymarl), [RLlib](imp_marl/imp_wrappers/examples/rllib/rllib_example.py), [MARLlib](imp_marl/imp_wrappers/marllib/marllib_wrap_ma_struct.py) etc.

## Acknowledgements 🙏

This repository is inspired by the following projects:

[PureJAXRL](https://github.com/luchris429/purejaxrl) by [luchris429](https://github.com/luchris429)

[CleanRL](https://github.com/vwxyzjn/cleanrl) started by [vwxyzjn](https://github.com/vwxyzjn)
