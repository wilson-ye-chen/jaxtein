import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev

class SteinKernel:
    def __init__(self, logp, dlogp, k):
        self.logp = logp
        self.dlogp = dlogp
        self.k = k
        self.dkx = jit(jacrev(k, argnums=0))
        self.dky = jit(jacrev(k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

    def evaluate(self, x, y):
        dkx = self.dkx(x, y)
        dky = self.dky(x, y)
        d2k = jnp.trace(self.d2k(x, y))
        k = self.k(x, y)
        dlogpx = self.dlogp(x)
        dlogpy = self.dlogp(y)
        k0 = d2k + \
            jnp.dot(dkx, dlogpy) + \
            jnp.dot(dky, dlogpx) + \
            k * jnp.dot(dlogpx, dlogpy)
        return k0

class KernelSteinDiscrepancy:
    def __init__(self, k0):
        self.k0 = k0
        self.vfk0 = jit(vmap(k0.evaluate, 0, 0))

    def k0mat(self, x):
        n = x.shape[0]
        ir, ic = np.tril_indices(n, k=-1)
        k0_tril = self.vfk0(x[ir], x[ic])
        k0_diag = self.vfk0(x, x)
        return (k0_tril, k0_diag, ir, ic)

    def evaluate(self, x):
        k0_tril, k0_diag, _, _ = self.k0mat(x)
        k0_sum = np.sum(k0_tril) * 2 + np.sum(k0_diag)
        return np.sqrt(k0_sum) / x.shape[0]

    def cumeval(self, x):
        n = x.shape[0]
        ksd = np.empty(n)
        k0_tril, k0_diag, ir, _ = self.k0mat(x)
        k0_sum = 0.
        for i in range(n):
            k0_sum += np.sum(k0_tril[ir == i]) * 2 + k0_diag[i]
            ksd[i] = np.sqrt(k0_sum) / (i + 1)
        return ksd
