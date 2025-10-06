# JAX vs NumPy

## JAX vs NumPy Performance Comparison ⚡

### Environment Rollouts

We compare the performance of JAX with NumPy (multiprocessing) for simulating rollouts for a k-out-of-n system with 5 components (agents), where each episode consists of 50 time steps. For 10,000 episodes, JAX (solid lines) achieves up to **~14x** speedup over NumPy (dashed lines).

![Runtime and Speedup vs NumPy](profile/jax_vs_numpy.svg)

| # Episodes | NumPy (for loop) [s] | NumPy (mp) [s] | JAX (scan) [s] | Speedup: NumPy (mp) vs. JAX (scan) |
|:-----------:|---------------------:|---------------:|-------------------:|---------------:|
| 1           | 0.01 | 1.18 | 0.27 | 4.39× |
| 10          | 0.05 | 1.2 | 0.24 | 5.01× |
| 100         | 0.51 | 1.3 | 0.24 | 5.41× |
| 1,000       | 5.09 | 2.09| 0.28 | 7.4× |
| 10,000      | 51.94 | 10.02  | 0.72 | **13.88×** |

### RL Training

We further benchmark RL training on variants of the k-out-of-n system. JAX achieves over **5x - 12x faster** training throughput than the equivalent PyTorch implementation running on 8 CPU cores.

| Environment | Agents | Episodes | Timesteps | MBP (JAX) [s] | MBP (PyTorch 8 CPUs) [s] | Speedup |
|:-------------|:-------:|:----------:|:-----------:|---------------:|--------------------------:|:--------:|
| `k_n_infinite` | 4 | 50,000 | 2.5 M | 403.9 s (0:06:44) | 4883.6 s (1:21:24) | **12×** |
| `kn_50` | 5 | 100,000 | 5 M | 1781 s (0:29:41) | 9571.8 s (2:39:32) | **5.4×** |

Hardware specs: Apple M1 Pro, 16GB RAM