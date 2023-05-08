import jax.numpy as jnp

def k_imq(x, y, cinv):
    z = x - y
    n2 = jnp.dot(jnp.dot(z, cinv), z)
    return (1 + n2) ** (-0.5)

def k_kgm(x, y, s, l, cinv):
    n2 = lambda z: jnp.dot(jnp.dot(z, cinv), z)
    t = (s - 1) / 2
    x = x - l
    y = y - l
    k = (1 + n2(x)) ** t * (1 + n2(y)) ** t * (1 + n2(x - y)) ** (-0.5) + \
        (1 + jnp.dot(jnp.dot(x, cinv), y)) / \
        ((1 + n2(x)) ** 0.5 * (1 + n2(y)) ** 0.5)
    return k
