import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev
from tqdm import tqdm

class SteinThinning:
    def __init__(self, k0):
        self.k0 = k0
        self.vfk0 = jit(vmap(k0.evaluate, 0, 0))

    def thin(self, x, m):
        # Pre-allocate arrays
        n = x.shape[0]
        k0m = np.empty((n, m))
        idx = np.empty(m, dtype=np.uint32)

        # Populate columns of k0m as new points are selected
        k0m[:, 0] = self.vfk0(x, x)
        idx[0] = np.argmin(k0m[:, 0])
        for i in tqdm(range(1, m)):
            x_last = np.tile(x[idx[i - 1]], (n, 1))
            k0m[:, i] = self.vfk0(x, x_last)
            idx[i] = np.argmin(k0m[:, 0] + 2 * np.sum(k0m[:, 1:(i + 1)], axis=1))
        return idx
