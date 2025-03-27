# Agents

**Core idea**: To train an agent, we defined a *pure* function `update_runner` that takes a `runner` state as input and returns a new `runner` state and a set of metrics. The `runner` contains all the stateful components of the agent, environment, total_timesteps, etc. required for training. Instead of using a for loop, we use `lax.scan` to iterate over the `update_runner` function for a fixed number of time steps `num_steps`:

```python
runner = init_runner(key)
runner, metrics = lax.scan(update_runner, runner, length=num_steps)
```

The `lax.scan` is functional programming construct that allows jit-compilation, and is much faster than using a for loop.

#### Example: DDQN Agent

The `runner` for a DDQN agent can simply be defined as a tuple as shown below. At each timestep, the `update_runner` is called. In this we use the `q_params` in the `runner` for selecting an action, `env_state` for stepping in the environment. Collected experience is used to update the `replay_buffer_state`. Sample from the `replay_buffer_state` to perform the TD-learning. We periodically update `target_params`, evaluate the agent. In the example below, we run the `update_runner` for `10_000` timesteps.

```python

class DDQN:

    def __init__(self):
        # initialise replay buffer, Q-network, target network, etc.
    
    def init_runner(self, key):
        # initialise runner state
        runner = (q_params, target_params, replay_buffer_state, env_state, total_timesteps)
        return runner

    def select_action(self, state):
        # select action using epsilon-greedy policy

    def update_replay_buffer(self, state, action, reward, next_state, tt):
        # update replay buffer

    def learning_phase(self, runner):
        # sample batch from replay buffer
        # compute loss
        # update Q-network

    def evaluate_and_checkpoint(self, runner):
        # evaluate agent
        # save checkpoint

    def update_runner(runner):
        # 1. select action
        # 2. step in the environment
        # 3. update replay buffer
        # (udpate runner state)
        # 4. learning phase
        # (udpate runner state)
        # 5. update target network
        # 6. evaluate agent and chkpt
        # 7. compute metrics
        return runner, metrics

    def train(self, key):
        runner = self.init_runner(key)
        runner, metrics = lax.scan(update_runner, runner, length=10_000)
        return runner, metrics
```
