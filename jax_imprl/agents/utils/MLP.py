from typing import Sequence

import jax.numpy as jnp

import flax.linen as nn


class MLP(nn.Module):
    architecture: Sequence[int]

    @nn.compact
    def __call__(self, x: jnp.array):
        for i, units in enumerate(self.architecture):
            x = nn.Dense(units, kernel_init=nn.initializers.orthogonal())(x)
            if i < len(self.architecture) - 1:
                x = nn.relu(x)
        return x
