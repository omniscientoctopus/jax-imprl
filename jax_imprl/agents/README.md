# Agents

- Core idea: The agent is defined as a function that takes a 'runner' state
as input and returns a new 'runner' state and a set of metrics.

- runner contains all the stateful components of the agent 
(network parameters, replay buffer, environment state, timestep, etc.)

1. Define a function that updates the runner state

```python
def update_runner(runner):

    # 1. select action
    # 2. step in the environment
    # 3. update replay buffer
    # (udpate runner state)
    # 4. learning phase
    # (udpate runner state)
    # 5. update target network
    # 6. compute metrics

    return runner, metrics
```

2. Define a training loop using lax.scan

```python
def train(key):
    runner, metrics = lax.scan(update_runner, runner, length=10_000)
    return runner, metrics
```
