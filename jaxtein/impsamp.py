import numpy as np
from qpsolvers import solve_qp
from jaxtein.stein import KernelSteinDiscrepancy
from jaxtein.posdef import nearestPD, isPD

class SteinImportanceSampling:
    def __init__(self, k0):
        self.k0 = k0
        self.ksd = KernelSteinDiscrepancy(k0)

    def k0full(self, x):
        n = x.shape[0]
        i = range(n)
        k0l, k0d, ir, ic = self.ksd.k0mat(x)
        k0f = np.zeros((n, n))
        k0f[ir, ic] = k0l
        k0f[ic, ir] = k0l
        k0f[i, i] = k0d
        return k0f

    def solve(self, x):
        # Remove duplicates
        x = np.unique(x, axis=0)
        n = x.shape[0]

        # Build cost matrix
        k0f = self.k0full(x)
        if not isPD(k0f):
            k0f = nearestPD(k0f)

        # Without linear term
        q = np.zeros(n)

        # Positive weights (gw <= h)
        g = -np.eye(n)
        h = np.zeros(n)

        # Sum to one (aw = b)
        a = np.ones(n)
        b = np.array([1.])

        # Solve
        w = solve_qp(k0f, q, g, h, a, b, solver='proxqp')
        return (w, x, k0f)

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
