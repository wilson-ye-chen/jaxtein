import numpy as np
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

class PiDistribution:
    def __init__(self, logp, dlogp, d2logp, k0):
        self.logp = logp
        self.dlogp = dlogp
        self.d2logp = d2logp
        self.k0 = k0
        self.k0xx = lambda x: k0.evaluate(x, x)
        self.dk0xx = jit(jacfwd(self.k0xx))

    def logpdf(self, x):
        return self.logp(x) + 0.5 * jnp.log(self.k0xx(x))

    def dlogpdf(self, x):
        return self.dlogp(x) + 0.5 * self.dk0xx(x) / self.k0xx(x)
