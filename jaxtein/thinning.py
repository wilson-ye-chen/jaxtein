import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import fori_loop

class SteinThinning:
    def __init__(self, k0):
        self.k0 = k0
        self.vfk0 = jit(vmap(k0.evaluate, 0, 0))

    def thin(self, x, m):
        n = x.shape[0]
        k0s = jnp.zeros(n)
        idx = jnp.zeros(m, dtype=jnp.int32)
        ksd = jnp.zeros(m)

        k0d = self.vfk0(x, x)
        idx = idx.at[0].set(jnp.argmin(k0d))
        ksd = ksd.at[0].set(k0d[idx[0]])

        def add_idx(i, val):
            k0s, idx, ksd = val
            x_last = jnp.tile(x[idx[i - 1]], (n, 1))
            k0s += self.vfk0(x, x_last)
            tmp = k0d + 2 * k0s
            idx = idx.at[i].set(jnp.argmin(tmp))
            ksd = ksd.at[i].set(tmp[idx[i]])
            return (k0s, idx, ksd)

        k0s, idx, ksd = fori_loop(1, m, add_idx, (k0s, idx, ksd))
        return (idx, ksd)
